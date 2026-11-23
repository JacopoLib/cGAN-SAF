# Training

> **Implementation map**
>
> | Concept | Code |
> |---|---|
> | Entry point | `train_final.py` |
> | Final training, checkpoint, bundles | `run_final_training`, `train_with_checkpoint`, `save_bundle` (`cgan/final_training.py`) |
> | Training loop | `fit_final` (`cgan/trainer.py`) |
> | Epoch: batches, critic and generator updates | `CGAN.train_epoch` |
> | Losses | `CGAN.critic_loss`, `CGAN.generator_loss` |
> | Metrics (CRPS, coverage, ...) | `CGAN.evaluate`, `crps_ensemble`, `central_interval` |
> | Batches, determinism, checkpoint | `cgan/utils.py` |

This page describes the final training of the three models: each model is trained on **the whole database(s)**, without a validation set,
with the hyperparameters of `configs/<model>.csv`.

## Command

```bash
python train_final.py                    # tfz, dcn and dcn_tfz, one after the other
python train_final.py --models dcn_tfz   # one model (tfz, dcn or dcn_tfz)
python train_final.py --descriptor mol2vec --out results/mol2vec
```

| Option | Default | Meaning |
|---|---|---|
| `--models` | all three | models to train |
| `--descriptor` | `config.DESCRIPTOR` (`mordred`) | `mordred` or `mol2vec` |
| `--out` | `results/final` | output folder; each model writes to `<out>/<model>/` (`<out>/<model>_mol2vec/` with Mol2Vec) |

For each model the script: loads the hyperparameters, enables determinism with `config.SEED`, builds the dataset and calls
`run_final_training`.

## One epoch

For every mini-batch of `batch_size` samples:

1. **Expansion.** Each sample is repeated `K = 32` times (`K_SAMPLES`), each copy with its own latent vector `z ~ N(0, I)`, so that the
   losses can compare the target with 32 generated values.
2. **Critic updates.** `k1` updates of the critics, with the mixture representation detached.
3. **Generator updates.** `k2` updates of the mixture encoder, the generator and the latent encoder; the mixture encoder is re-run, so that each step has its own graph.

`(k1, k2)` follows from `k1_over_k2`: `(int(r), 1)` if `r >= 1` and `(1, int(1/r))` otherwise. `k1_over_k2 = 2` gives two critic updates
per generator update; `0.5` gives one critic update per two generator updates.

The critics and the generator side use two Adam optimisers (learning rates `10**lr_T` and `10**lr_G`, see
[Configuration reference](Configuration-reference.md)). The gradient norm is clipped to 10 for the critics and for each of the mixture encoder, generator and
latent encoder.

### Batches

- **Single-task models**: random batches of the whole database, reshuffled every epoch, and the **last incomplete batch is dropped** (`drop_last`): the latent term of the loss is computed over the batch
  and is unreliable on a very small batch.
- **Multi-task model**: balanced batches with the same number of samples of each task, `max(1, batch_size // n_tasks)` per task. The number of
  batches per epoch is set by the task with fewer samples (the DCN database: 929 samples against 1154), so an epoch does not visit all the samples
  of the larger database; the shuffling changes every epoch, so all the samples are used over the epochs.

## Losses

Let `y` be the target (z-scored), `y_hat` the generated values and `K = 32`. Tasks without a label in a sample are ignored (NaN-aware losses); the
losses are averaged over the tasks that have labels in the batch.

**Critic (`critic_loss`)**, per task:

```
L_T = - [ mean log(1 - sigmoid(T(x, y)))  +  mean log(sigmoid(T(x, y_hat.detach()))) ]
```

The critic is trained to give a low score to real pairs and a high score to generated pairs.

**Generator (`generator_loss`)**, per task `t`:

```
L_G,t = softplus(T_t(x, y_hat_t)).mean()                     adversarial, non-saturating
      + (1 - lam) * log_q                                     latent-space regularisation
      + beta * ( softmin_t + alpha_mean_t * (mean_k y_hat - y)^2 ).mean()   reconstruction
```

and the total is the mean over the tasks (`nanmean`).

- **Adversarial term**: the generator is trained to obtain a low score, as the real pairs.
- **Latent term** `log_q = 0.5 * sum_d (1 + log var_d - mu_d^2 - var_d)`, where `mu` and `var` are the mean and the variance, over the batch, of the output
  of the latent encoder `net_E([x, y_hat])` (dimension `n_f`). It equals `-KL(N(mu, var) || N(0, I))`, so with `lam > 1` the term
  `(1 - lam) * log_q` is a KL penalty of weight `lam - 1` (all the final models use `lam = 1.5`, a weight of 0.5).
- **Softmin-of-K reconstruction**: for the `K` generated values `y_hat_k` of a sample,
  `softmin = -tau * logsumexp(-(y_hat_k - y)^2 / tau) + tau * log K`. For a small `tau` it approaches the squared error of the *best* of the `K`
  samples: the generator is asked to cover the target with some of its samples, not to concentrate all of them on it. The term `alpha_mean * (mean_k y_hat_k - y)^2`
  anchors the mean of the samples to the target. `tau` and `alpha_mean` are per task.
- `beta` multiplies the reconstruction term.

## Metrics

At the end of every epoch, `CGAN.evaluate` runs on the training samples (batches of `batch_size`, a fixed random order, `K = 32` generated values per sample) and computes, per task:

| Metric | Meaning |
|---|---|
| `crps` | ensemble CRPS in z-scored units, `mean|y_k - y| - 0.5 mean|y_k - y_l|` over the `K` samples |
| `crps_orig` | the same in physical units |
| `rmse`, `mae`, `r2` | errors of the generated values against the target, `r2` against the mean of the targets of the batch |
| `cov95`, `cov50` | fraction of targets inside the central 95% and 50% interval of the generated values |
| `width95` | mean width of the 95% interval, physical units |
| `loss_G`, `loss_T` | generator and critic losses |

## Learning-rate schedule

The learning rates of both optimisers are halved (`ReduceLROnPlateau`, patience 10, factor 0.5, threshold 0) when the mean generator loss
(for the generator optimiser) or the mean critic loss (for the critic optimiser) of the epoch has not decreased for 10 epochs. The loss
of a mini-batch is that of the last update of the batch.

## Checkpoint and stopping

After every epoch the CRPS in z-scored units, **averaged over the tasks and computed on the training data**, is compared with the best one so far.
When it is lower, the weights of all the networks are saved to `best_global.pt`. Training stops after `MAX_EPOCHS = 1000` epochs, or after
`PATIENCE = 100` epochs without improvement. The weights of the best epoch are then loaded back.

Because the model is trained on the whole database, this CRPS is a training-set score. It is a criterion to choose the epoch, not an estimate of the
error on new data.

## Reproducibility

`train_final.py` calls `enable_determinism(SEED = 42)` before building the dataset: it seeds Python, NumPy and PyTorch, sets cuDNN to deterministic mode,
disables its autotuner, sets `CUBLAS_WORKSPACE_CONFIG` and `torch.use_deterministic_algorithms(True)`. The seed is set again before training (a `seed`
column in the configuration file overrides `SEED`), and the seed of the epoch `e` is `seed + e`. The result also depends on the PyTorch version, on the hardware (CPU or GPU) and, for the descriptors, on the RDKit
version: bitwise identical weights should not be expected on a different machine.

## Outputs

`<out>/<model>/` contains:

| File | Content |
|---|---|
| `best_global.pt` | weights of the best epoch (all the networks, plus the epoch and the score) |
| `bundle_<task>.pt` | inference bundle of each task: one for the single-task models, two for `dcn_tfz` (`bundle_dcn.pt`, `bundle_tfz.pt`) |
| `history.csv` | per epoch: `epoch`, `crps`, `rmse` (means over the tasks), `loss_G`, `loss_T` |
| `config.csv` | the hyperparameters used |
| `predictions_<task>.csv` | `fuel`, `measured`, `mean`, `std`, `q2.5`, `q97.5` for every labelled training sample (1000 Monte Carlo draws, physical units; replicates of a compound appear in separate rows) |

The bundle (`save_bundle`) is a PyTorch file with: the hyperparameters, the task id and names, the gates, the descriptor kind and its information
(`descriptor_info`: columns and medians), the species vocabulary and its standardised descriptors, the descriptor mean and standard deviation, the target
mean and standard deviation, and the state of the mixture encoder and of the generator. It is self-contained: `predict.py` needs nothing else.

The progress is printed at the end of the run only (best CRPS and epoch); follow `history.csv` while the run is going.
