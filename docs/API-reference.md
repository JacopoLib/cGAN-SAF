# API reference

The package `cgan/` and the two scripts at the root. Shapes use `B` for the batch, `L` for the number of descriptors, `T` for the number of tasks.

## Scripts

| Script | Role |
|---|---|
| `train_final.py` | final training of the models in `config.MODELS`; options `--models`, `--descriptor`, `--out` |
| `predict.py` | `Predictor` class and command line for predictions from a bundle |

### `predict.py`

| Name | Description |
|---|---|
| `Predictor(bundle_path, device="cpu")` | rebuilds the mixture encoder and the generator of one task from a bundle; builds the canonical-SMILES index of the training species |
| `Predictor.sample(compositions, n_samples=1000, seed=42)` | `compositions`: list of `{SMILES: fraction}`; returns `(n_fuels, n_samples)` samples in physical units |
| `Predictor._repository_smiles(smiles)` | the training-vocabulary spelling of a known molecule (ValueError for an invalid SMILES) |
| `read_compositions(args)` | identifiers and compositions from `--smiles` or `--input` |

## `cgan/config.py`

Constants only; see [Configuration reference](Configuration-reference.md). The function `descriptor_selection(tasks)` returns the Mordred column selection
(`fixed` or `filter`) of the model whose tasks are `tasks`.

## `cgan/data.py`

| Name | Description |
|---|---|
| `load_dcn()`, `load_tfz()` | `[(fuel id, value, {SMILES: fraction})]` for pure compounds and blends of one property |
| `LOADERS` | task name to loader |
| `species_vocabulary(tasks)` | distinct SMILES of the databases of `tasks`, in order of appearance |
| `Dataset` | dataclass: `tasks`, `fuels`, `y` (z-scored, NaN where the task has no label), `task_id`, `comp_idx`, `comp_frac`, `species`, `Z_species`, `desc_mean`, `desc_std`, `y_mean`, `y_std`, `blend_gate`, `descriptor`, `descriptor_info`; properties `n_tasks`, `latent_dim`; method `denormalize(y_norm, task)` |
| `build_dataset(tasks, descriptor)` | assembles the `Dataset` of the requested tasks |

## `cgan/descriptors.py`

| Name | Description |
|---|---|
| `clean_smiles(smiles)` | removes all whitespace |
| `compute_mordred(smiles)` | raw Mordred table (1D, 2D, 3D) of a list of SMILES |
| `mordred_raw(smiles, cache_path, recompute, update_cache)` | raw table from the cache, computing the missing species |
| `mordred_columns(path)` | the 347 names of the fixed selection |
| `filter_descriptors(X)` | filtered, median-imputed matrix and mask of the kept columns |
| `mordred_descriptors(smiles, selection, ...)` | descriptor matrix and `info` (columns, medians) |
| `mordred_transform(smiles, info)` | descriptors of new species, restricted to the columns of `info` |
| `mol2vec_descriptors(smiles, model_path)` | 300-dimensional Mol2Vec embeddings |
| `species_descriptors(smiles, kind, recompute, selection)` | descriptor matrix and `info` for `mordred` or `mol2vec` |
| `new_species_descriptors(smiles, kind, info)` | raw descriptors of species outside the vocabulary |

Run as a module (`python -m cgan.descriptors`) to build the cache: options `--tasks`, `--kind`, `--recompute`.

## `cgan/models.py`

| Class | Description |
|---|---|
| `MLP` | Linear + LayerNorm + activation input layer, hidden layers (plain or residual), linear output; optional spectral normalisation |
| `ResidualBlock` | `x + fc2(dropout(act(fc1(x))))` |
| `Critic` | discriminator of one task, scores `(x, y)` |
| `SharedTrunk` | backbone shared by the heads of the multi-task generator |
| `BridgeNet` | generator head of one task |
| `Generator` | one `BridgeNet` per task, optional `SharedTrunk`; `forward(x, z)` returns `(B, T)` |
| `MultiQueryPooling` | attention pooling with learnable queries and a volume-fraction bias |
| `MixtureEncoder` | `forward(comp_idx, comp_frac, Z_species, task_id)` returns `(mean_pool, blend_delta)` |

## `cgan/trainer.py`

| Name | Description |
|---|---|
| `HyperParameters` | dataclass built from a row of `configs/<model>.csv` with `from_config(cfg, ds)`; property `update_counts` |
| `CGAN` | networks, optimisers and schedulers; `encode`, `critic_loss`, `generator_loss`, `train_epoch`, `evaluate`, `state`, `load` |
| `fit_final(ds, hp, rows, seed, device, max_epochs, patience, checkpoint)` | training loop with the checkpoint on the CRPS; returns the model, the metrics of the best epoch and the history |
| `crps_ensemble`, `central_interval`, `masked_error`, `expand` | metric and helper functions |

## `cgan/final_training.py`

| Name | Description |
|---|---|
| `load_config(name)` | row of `configs/<name>.csv` as a dictionary |
| `train_with_checkpoint(ds, cfg, rows, ckpt_path)` | training on `rows`; loads back the best weights |
| `predict(model, rows, n_samples)` | Monte Carlo samples `(len(rows), n_tasks, n_samples)` in physical units |
| `save_predictions(ds, model, rows, out_dir)` | `predictions_<task>.csv` |
| `run_final_training(ds, cfg, out_dir)` | the whole procedure of one model: training, history, predictions, bundles |
| `save_bundle(ds, model, cfg, task, path)` | inference bundle of one task |

## `cgan/utils.py`

`enable_determinism(seed)`, `set_seed(seed)`, `FixedBatchSampler`, `balanced_batches(task_of_row, batch_size, n_tasks, seed)`, `Checkpointer(patience, path)`.
