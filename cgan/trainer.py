"""c-GAN training: networks, adversarial losses, softmin-of-K reconstruction loss, validation metrics and `fit`."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .models import MLP, Critic, Generator, MixtureEncoder
from .utils import Checkpointer, FixedBatchSampler, balanced_batches

K_SAMPLES = 32  # latent samples drawn per data point during training and validation
MAX_GRAD_NORM = 10.0


@dataclass
class HyperParameters:
    """Hyperparameters of one c-GAN (see Table 2 of the paper and Table S4-1 of the Supplementary Material)."""

    batch_size: int
    lr_G: float  # generator / encoder learning rate
    lr_T: float  # discriminator learning rate
    num_queries: int
    head_layers: tuple  # residual blocks of each generator head: (stage 1, stage 2)
    head_widths: tuple  # width of each generator head: (stage 1, stage 2)
    shared: tuple  # (width, blocks) of the shared backbone, None for single-task models
    critic_layers: int
    critic_width: int
    encoder_layers: int
    encoder_width: int
    features: int  # latent dimension
    activation: str
    update_ratio: float  # discriminator updates per generator update (k1 / k2)
    lam: float
    beta: float
    dropout: float
    alpha_mean: list  # per task: weight of the mean-anchor term
    tau: list  # per task: softmin-of-K temperature
    blend_gate: list  # per task: gate the attention correction with the is-blend indicator (beta_t)

    @classmethod
    def from_config(cls, cfg, ds):
        """Build from a flat config dict (a row of configs/<model>.csv) for the tasks of `ds`."""
        n_tasks = ds.n_tasks

        def per_task(name):
            return (
                [float(cfg[f"{name}_{t}"]) for t in range(n_tasks)]
                if f"{name}_0" in cfg
                else [float(cfg[name])] * n_tasks
            )

        shared = (int(round(cfg["neurons_shared"])), int(round(cfg["num_layers_shared"]))) if n_tasks > 1 else None
        return cls(
            batch_size=int(round(cfg["batch_size"])),
            lr_G=10.0 ** cfg["lr_G"],
            lr_T=10.0 ** cfg["lr_T"],
            num_queries=int(round(cfg["num_queries"])),
            head_layers=(int(round(cfg["num_layers_trunk"])), int(round(cfg["num_layers_D"]))),
            head_widths=(int(round(cfg["neurons_trunk"])), int(round(cfg["neurons_D"]))),
            shared=shared,
            critic_layers=int(round(cfg["num_layers_T"])),
            critic_width=int(round(cfg["neurons_T"])),
            encoder_layers=int(round(cfg["num_layers_E"])),
            encoder_width=int(round(cfg["neurons_E"])),
            features=int(cfg["features"]),
            activation=cfg["activation"],
            update_ratio=float(cfg["k1_over_k2"]),
            lam=float(cfg["lam"]),
            beta=float(cfg["beta"]),
            dropout=float(cfg["dropout"]),
            alpha_mean=per_task("alpha_mean"),
            tau=per_task("tau"),
            blend_gate=list(ds.blend_gate),
        )

    @property
    def update_counts(self):
        """(critic updates k1, generator updates k2) per mini-batch."""
        r = self.update_ratio
        return (int(r), 1) if r >= 1 else (1, int(1 / r))


# ---------------------------------------------------------------------------------------------------------------------
# Metrics (all NaN-aware: tasks without a label in a sample are ignored)
# ---------------------------------------------------------------------------------------------------------------------


def masked_error(y_true, y_pred, fn):
    """Per-task mean of fn(y_true - y_pred) over labelled samples."""
    valid = ~torch.isnan(y_true)
    err = torch.where(valid, fn(y_true - y_pred), torch.zeros_like(y_true))
    return err.sum(dim=0) / (valid.sum(dim=0) + 1e-8)


def crps_ensemble(y_true, y_samples):
    """Ensemble CRPS, mean over samples. y_true: (B,), y_samples: (B, K)."""
    term1 = (y_samples - y_true.unsqueeze(1)).abs().mean(dim=1)
    term2 = 0.5 * (y_samples.unsqueeze(2) - y_samples.unsqueeze(1)).abs().mean(dim=(1, 2))
    return (term1 - term2).mean()


def central_interval(y_samples, level):
    """Lower/upper bounds of the central `level` interval of the samples. y_samples: (B, K)."""
    alpha = 1.0 - level
    return torch.quantile(y_samples, alpha / 2.0, dim=1), torch.quantile(y_samples, 1.0 - alpha / 2.0, dim=1)


def expand(t, K):
    """Repeat each row K times: (B, ...) -> (B*K, ...)."""
    return t.repeat_interleave(K, dim=0)


# ---------------------------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------------------------


class CGAN:
    """Mixture encoder, generator, one discriminator per task and latent-space encoder, with their optimizers."""

    def __init__(self, ds, hp, device):
        self.ds, self.hp, self.device = ds, hp, torch.device(device)
        L, T = ds.latent_dim, ds.n_tasks
        x_dim = 2 * L + config.N_AUX  # [mean_pool, attention correction, reserved columns]

        self.Z_species = ds.Z_species.to(self.device)
        self.mix_enc = MixtureEncoder(L, hp.num_queries, hp.blend_gate)
        self.net_D = Generator(
            T, hp.features, 2 * L, hp.head_widths, hp.head_layers, hp.activation, hp.dropout, hp.shared
        )
        self.net_Ts = nn.ModuleList([Critic(x_dim, hp.activation, hp.critic_layers, hp.critic_width) for _ in range(T)])
        self.net_E = MLP(x_dim + T, hp.features, hp.encoder_layers, hp.encoder_width, hp.activation, use_residual=True)
        for net in self.networks:
            net.to(self.device)

        self.opt_G = Adam(
            list(self.mix_enc.parameters()) + list(self.net_D.parameters()) + list(self.net_E.parameters()), lr=hp.lr_G
        )
        self.opt_T = Adam(self.net_Ts.parameters(), lr=hp.lr_T)
        plateau = dict(mode="min", patience=10, factor=0.5, threshold=0, threshold_mode="rel")
        self.sched_G = ReduceLROnPlateau(self.opt_G, **plateau)
        self.sched_T = ReduceLROnPlateau(self.opt_T, **plateau)
        self.y_mean = torch.tensor(ds.y_mean, device=self.device)
        self.y_std = torch.tensor(ds.y_std, device=self.device)

    @property
    def networks(self):
        return [self.mix_enc, self.net_D, self.net_Ts, self.net_E]

    def train(self, mode=True):
        for net in self.networks:
            net.train(mode)

    def state(self):
        return {
            "net_D": self.net_D.state_dict(),
            "net_E": self.net_E.state_dict(),
            "net_Ts": [T.state_dict() for T in self.net_Ts],
            "mix_enc": self.mix_enc.state_dict(),
        }

    def load(self, state):
        self.net_D.load_state_dict(state["net_D"])
        self.net_E.load_state_dict(state["net_E"])
        for T, s in zip(self.net_Ts, state["net_Ts"]):
            T.load_state_dict(s)
        self.mix_enc.load_state_dict(state["mix_enc"])

    def encode(self, rows):
        """Generator/critic input of dataset rows: [mean_pool, attention correction, 0, 0]."""
        ds = self.ds
        idx = torch.as_tensor(rows, dtype=torch.long)
        mean_pool, delta = self.mix_enc(
            ds.comp_idx[idx].to(self.device),
            ds.comp_frac[idx].to(self.device),
            self.Z_species,
            torch.as_tensor(ds.task_id[rows], dtype=torch.long, device=self.device),
        )
        aux = torch.zeros(mean_pool.shape[0], config.N_AUX, device=self.device)
        return torch.cat([mean_pool, delta, aux], dim=1)

    # --- losses ------------------------------------------------------------------------------------------------------

    def critic_loss(self, x, y, z):
        """Logistic GAN loss of the discriminators, averaged over the tasks labelled in the batch."""
        y_pred = self.net_D(x, z)
        total, n_used = 0.0, 0
        for t in range(y.shape[1]):
            valid = ~torch.isnan(y[:, t])
            if valid.sum() == 0:
                continue
            real = self.net_Ts[t](x, torch.nan_to_num(y[:, t : t + 1], nan=0.0))
            fake = self.net_Ts[t](x, y_pred[:, t : t + 1].detach())
            total = total - (
                torch.log(1.0 - torch.sigmoid(real[valid]) + 1e-8).mean()
                + torch.log(torch.sigmoid(fake[valid]) + 1e-8).mean()
            )
            n_used += 1
        if n_used == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        return total / n_used

    def generator_loss(self, x, y, z, K):
        """Non-saturating adversarial term, latent-space regularization and softmin-of-K reconstruction loss.

        x, y, z hold K consecutive rows per data point (see `expand`)."""
        hp, n_tasks = self.hp, y.shape[1]
        y_pred = self.net_D(x, z)

        adversarial = torch.full((n_tasks,), torch.nan, device=x.device)
        for t in range(n_tasks):
            valid = ~torch.isnan(y[:, t])
            if valid.sum() > 0:
                adversarial[t] = F.softplus(self.net_Ts[t](x, y_pred[:, t : t + 1])[valid]).mean()

        y_for_E = y_pred * torch.isfinite(y).float()
        z_enc = self.net_E(torch.cat((x, y_for_E), dim=1))
        mu, var = z_enc.mean(dim=0), z_enc.var(dim=0, unbiased=False)
        log_q = 0.5 * torch.sum(1 + torch.log(var) - mu**2 - var)  # -KL(q(z|x,y) || N(0, I))

        B = y_pred.shape[0] // K
        y_pred_BKY, y_true_BY = y_pred.view(B, K, n_tasks), y.view(B, K, n_tasks)[:, 0, :]
        recon = torch.full((n_tasks,), torch.nan, device=x.device)
        for t in range(n_tasks):
            valid = ~torch.isnan(y_true_BY[:, t])
            if valid.sum() == 0:
                continue
            y_true, y_samp = y_true_BY[valid, t], y_pred_BKY[valid, :, t]
            sq_err = (y_samp - y_true.unsqueeze(1)) ** 2
            tau = hp.tau[t]
            softmin = -tau * torch.logsumexp(-sq_err / tau, dim=1) + tau * np.log(K)
            mean_anchor = (y_samp.mean(dim=1) - y_true) ** 2
            recon[t] = (softmin + hp.alpha_mean[t] * mean_anchor).mean()

        loss_per_task = adversarial + (1.0 - hp.lam) * log_q + hp.beta * recon
        return torch.nanmean(loss_per_task)

    # --- training and validation -------------------------------------------------------------------------------------

    def train_epoch(self, rows, seed):
        """One epoch over `rows` (dataset row indices); returns the mean generator and discriminator losses over the
        mini-batches (last update of each)."""
        ds, hp = self.ds, self.hp
        data = TensorDataset(torch.as_tensor(ds.y[rows], dtype=torch.float32), torch.as_tensor(rows, dtype=torch.long))
        g = torch.Generator()
        g.manual_seed(seed)
        if ds.n_tasks == 1:
            loader = DataLoader(data, batch_size=hp.batch_size, shuffle=True, generator=g)
        else:
            batches = balanced_batches(ds.task_id[rows], hp.batch_size, ds.n_tasks, seed)
            loader = DataLoader(
                data, batch_sampler=FixedBatchSampler(batches, shuffle_batches=True, seed=seed), generator=g
            )

        k1, k2 = hp.update_counts
        self.train()
        loss_G = loss_T = 0.0
        for y_batch, row_ids in loader:
            row_ids = row_ids.numpy()
            y = expand(y_batch.to(self.device), K_SAMPLES)
            z = torch.randn(len(row_ids), K_SAMPLES, hp.features, device=self.device).view(-1, hp.features)
            x = expand(self.encode(row_ids), K_SAMPLES)

            for _ in range(k1):
                self.opt_T.zero_grad(set_to_none=True)
                critic_loss = self.critic_loss(x.detach(), y, z)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.net_Ts.parameters(), max_norm=MAX_GRAD_NORM)
                self.opt_T.step()

            for _ in range(k2):
                self.opt_G.zero_grad(set_to_none=True)
                # the mixture encoder is re-run so that each generator step has its own graph
                generator_loss = self.generator_loss(expand(self.encode(row_ids), K_SAMPLES), y, z, K_SAMPLES)
                generator_loss.backward()
                for net in (self.mix_enc, self.net_D, self.net_E):
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=MAX_GRAD_NORM)
                self.opt_G.step()
            loss_T += float(critic_loss)
            loss_G += float(generator_loss)
        return loss_G / len(loader), loss_T / len(loader)

    @torch.no_grad()
    def evaluate(self, rows):
        """Validation metrics on `rows`: per-task arrays for rmse, r2, mae, crps (normalized units), coverage and
        interval width (physical units), and the mean generator/discriminator losses."""
        ds, hp, T = self.ds, self.hp, self.ds.n_tasks
        self.train(False)
        g = torch.Generator()
        g.manual_seed(config.SEED)
        order = torch.randperm(len(rows), generator=g).numpy()
        acc = {k: np.zeros(T) for k in ("rmse", "r2", "mae")}
        sums = {k: np.zeros(T) for k in ("crps", "crps_orig", "cov95", "cov50", "width95")}
        count = np.zeros(T)
        loss_G = loss_T = 0.0
        n_batches = 0

        for start in range(0, len(rows), hp.batch_size):
            batch_rows = rows[order[start : start + hp.batch_size]]
            K = K_SAMPLES
            y = expand(torch.as_tensor(ds.y[batch_rows], dtype=torch.float32, device=self.device), K)
            z = torch.randn(len(batch_rows), K, hp.features, device=self.device).view(-1, hp.features)
            x = expand(self.encode(batch_rows), K)

            loss_T += float(self.critic_loss(x, y, z))
            loss_G += float(self.generator_loss(x, y, z, K))
            y_pred = self.net_D(x, z)

            mse = masked_error(y, y_pred, torch.square)
            baseline = masked_error(y, torch.nanmean(y, dim=0), torch.square)
            acc["rmse"] += mse.sqrt().cpu().numpy()
            acc["r2"] += (1 - mse / (baseline + 1e-8)).cpu().numpy()
            acc["mae"] += masked_error(y, y_pred, torch.abs).cpu().numpy()
            n_batches += 1

            B = len(batch_rows)
            y_true_BY = y.view(B, K, T)[:, 0, :]
            y_samp_BKY = y_pred.view(B, K, T)
            for t in range(T):
                valid = ~torch.isnan(y_true_BY[:, t])
                n = int(valid.sum())
                if n == 0:
                    continue
                y_true, y_samp = y_true_BY[valid, t], y_samp_BKY[valid, :, t]
                y_true_o, y_samp_o = y_true * self.y_std[t] + self.y_mean[t], y_samp * self.y_std[t] + self.y_mean[t]
                lo95, hi95 = central_interval(y_samp_o, 0.95)
                lo50, hi50 = central_interval(y_samp_o, 0.50)
                for key, value in (
                    ("crps", crps_ensemble(y_true, y_samp)),
                    ("crps_orig", crps_ensemble(y_true_o, y_samp_o)),
                    ("cov95", ((y_true_o >= lo95) & (y_true_o <= hi95)).float().mean()),
                    ("cov50", ((y_true_o >= lo50) & (y_true_o <= hi50)).float().mean()),
                    ("width95", (hi95 - lo95).mean()),
                ):
                    sums[key][t] += float(value) * n
                count[t] += n

        out = {k: v / n_batches for k, v in acc.items()}
        out.update({k: v / np.maximum(count, 1.0) for k, v in sums.items()})
        out.update(loss_G=loss_G / n_batches, loss_T=loss_T / n_batches)
        return out


# ---------------------------------------------------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------------------------------------------------


def fit_final(ds, hp, rows, seed, device, max_epochs, patience, checkpoint):
    """Final training on all of `rows`, as in the paper's final models.

    Every epoch, the CRPS (mean over tasks, normalized units) is computed on the training data themselves; the weights
    of the epoch with the lowest CRPS are saved to `checkpoint`, training stops after `patience` epochs without
    improvement, and the learning-rate schedulers follow the training losses.
    Returns the model, the metrics of the best epoch and the per-epoch history.
    """
    model = CGAN(ds, hp, device)
    monitor = Checkpointer(patience, checkpoint)
    best, history = None, []
    for epoch in range(1, max_epochs + 1):
        loss_G, loss_T = model.train_epoch(rows, seed + epoch)
        metrics = model.evaluate(rows)
        model.sched_G.step(loss_G)
        model.sched_T.step(loss_T)

        crps = float(np.nanmean(metrics["crps"]))
        history.append(
            {
                "epoch": epoch,
                "crps": crps,
                "rmse": float(np.nanmean(metrics["rmse"])),
                "loss_G": loss_G,
                "loss_T": loss_T,
            }
        )
        stop, improved = monitor.step(crps, epoch, model.state)
        if improved:
            best = {**metrics, "epoch": epoch}
        if stop:
            break
    return model, best, history
