"""Final training with fixed hyperparameters, Monte Carlo predictions and inference bundles."""

import os

import numpy as np
import pandas as pd
import torch

from . import config
from .trainer import HyperParameters, fit_final
from .utils import set_seed


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(name):
    """Hyperparameters of one of the models in config.MODELS, as a flat dict."""
    return pd.read_csv(os.path.join(config.CONFIG_DIR, f"{name}.csv")).iloc[0].to_dict()


def train_with_checkpoint(ds, cfg, rows, ckpt_path):
    """Train on `rows` with the checkpoint on the training CRPS; the weights of the best epoch are loaded back."""
    seed = int(cfg.get("seed", config.SEED))
    set_seed(seed)
    hp = HyperParameters.from_config(cfg, ds)
    model, best, history = fit_final(ds, hp, rows, seed, device(), config.MAX_EPOCHS, config.PATIENCE, ckpt_path)
    model.load(torch.load(ckpt_path, map_location=device(), weights_only=False))
    return model, best, history


@torch.no_grad()
def predict(model, rows, n_samples=config.N_MC_SAMPLES):
    """Monte Carlo samples of the predictive distribution in physical units: (len(rows), n_tasks, n_samples)."""
    ds = model.ds
    model.train(False)
    x = model.encode(rows)
    out = np.empty((len(rows), ds.n_tasks, n_samples), dtype=np.float32)
    for n in range(n_samples):
        z = torch.randn(len(rows), model.hp.features, device=model.device)
        out[:, :, n] = (model.net_D(x, z) * model.y_std + model.y_mean).cpu().numpy()
    return out


def save_predictions(ds, model, rows, out_dir):
    """Per task: predictive mean, standard deviation and 95% interval next to the measured values (one row per sample)."""
    samples = predict(model, rows)
    for t, task in enumerate(ds.tasks):
        labelled = ~np.isnan(ds.y[rows, t])
        s = samples[labelled, t, :]
        pd.DataFrame(
            {
                "fuel": ds.fuels[rows[labelled]],
                "measured": ds.denormalize(ds.y[rows[labelled], t], t),
                "mean": s.mean(axis=1),
                "std": s.std(axis=1),
                "q2.5": np.quantile(s, 0.025, axis=1),
                "q97.5": np.quantile(s, 0.975, axis=1),
            }
        ).to_csv(os.path.join(out_dir, f"predictions_{task}.csv"), index=False)


def run_final_training(ds, cfg, out_dir):
    """Train on the whole dataset and write the weights, training history, predictions on the training data and one
    inference bundle per task."""
    os.makedirs(out_dir, exist_ok=True)
    rows = np.arange(len(ds.fuels))
    model, best, history = train_with_checkpoint(ds, cfg, rows, os.path.join(out_dir, "best_global.pt"))
    print(
        f"Best training CRPS (normalized units, mean over tasks): {np.nanmean(best['crps']):.4f}, epoch {best['epoch']}"
    )
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    pd.DataFrame([cfg]).to_csv(os.path.join(out_dir, "config.csv"), index=False)
    save_predictions(ds, model, rows, out_dir)
    for t, task in enumerate(ds.tasks):
        save_bundle(ds, model, cfg, t, os.path.join(out_dir, f"bundle_{task}.pt"))


def save_bundle(ds, model, cfg, task, path):
    """Self-contained inference bundle of one task: weights, hyperparameters, species descriptors and statistics."""
    torch.save(
        {
            "config": cfg,
            "task_id": int(task),
            "task_name": ds.tasks[task],
            "task_names": list(ds.tasks),
            "blend_gate": list(model.hp.blend_gate),
            "descriptor": ds.descriptor,
            "descriptor_info": ds.descriptor_info,
            "species": list(ds.species),
            "Z_species": ds.Z_species.cpu(),  # standardized descriptors of the training species
            "desc_mean": ds.desc_mean.cpu(),
            "desc_std": ds.desc_std.cpu(),
            "y_mean": ds.y_mean,
            "y_std": ds.y_std,
            "mix_enc_state": model.mix_enc.state_dict(),
            "net_D_state": model.net_D.state_dict(),
        },
        path,
    )
    print("Saved bundle:", path)
