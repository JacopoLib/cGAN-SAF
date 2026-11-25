# Configuration reference

There are two kinds of settings: the global settings in `cgan/config.py` and the hyperparameters of each model in `configs/<model>.csv`.

## `cgan/config.py`

| Name | Default | Meaning |
|---|---|---|
| `ROOT`, `DATA_DIR`, `DESCRIPTOR_DIR`, `CONFIG_DIR` | derived from the location of the file | paths of the repository, `data/`, `data/descriptors/`, `configs/` |
| `MOL2VEC_MODEL` | `data/descriptors/model_300dim.pkl` | path of the pre-trained Mol2Vec model; the environment variable `MOL2VEC_MODEL` overrides it |
| `DESCRIPTOR` | `"mordred"` | species descriptor: `"mordred"` or `"mol2vec"`; `--descriptor` on `train_final.py` overrides it |
| `TASKS` | `dcn`: `blend_gate=False`; `tfz`: `blend_gate=True` | per-task gate β<sub>t</sub> of the attention pooling (see [Model architecture](Model-architecture.md#mixture-encoder)) |
| `MODELS` | `tfz`: (`tfz`), `dcn`: (`dcn`), `dcn_tfz`: (`dcn`, `tfz`) | the models of the paper; the order of the tasks of a model is the order of the columns of the targets and of the heads |
| `N_AUX` | 2 | reserved input columns (temperature, "has temperature"), always zero |
| `SEED` | 42 | seed of the runs and of the Monte Carlo draws of `predict.py` |
| `MAX_EPOCHS` | 1000 | epoch budget of the final training |
| `PATIENCE` | 100 | epochs without improvement of the training CRPS before stopping |
| `N_MC_SAMPLES` | 1000 | Monte Carlo samples of the predictive distribution |

Two more constants are in `cgan/trainer.py`: `K_SAMPLES = 32` (latent samples per data point in the training losses and in the epoch metrics) and `MAX_GRAD_NORM = 10`
(gradient clipping).

## `configs/<model>.csv`

One header line and one row. `HyperParameters.from_config` reads it. Columns that end with `_0`, `_1` are per task, in the order of the tasks of the model
(`dcn_tfz`: 0 = DCN, 1 = freezing point); a column without suffix is used for all the tasks. Integers are rounded. The symbols are those of the paper (see [Notation](Notation.md)).

| Column | Symbol | Meaning |
|---|---|---|
| `lr_G`, `lr_T` | l<sub>r,𝒢/ℰ</sub>, l<sub>r,𝒟</sub> | **log10** of the learning rate of the generator side (mixture encoder, generator, latent encoder) and of the discriminators: `-3` is 1e-3 |
| `alpha_mean` or `alpha_mean_<t>` | α<sub>t</sub> | weight of the mean-anchor term of the reconstruction loss |
| `tau` or `tau_<t>` | τ<sub>t</sub> | temperature of the softmin over the `K = 32` samples |
| `num_queries` | Q | number of attention queries |
| `batch_size` | b | mini-batch size (multi-task: split equally between the tasks) |
| `num_layers_trunk`, `neurons_trunk` | N<sub>𝒢,1</sub>, m<sub>𝒢,1</sub> | residual blocks and width of stage 1 of each generator head |
| `num_layers_D`, `neurons_D` | N<sub>𝒢,2</sub>, m<sub>𝒢,2</sub> | residual blocks and width of stage 2 of each generator head |
| `num_layers_E`, `neurons_E` | N<sub>ℰ</sub>, m<sub>ℰ</sub> | residual blocks and width of the latent encoder |
| `num_layers_T`, `neurons_T` | N<sub>𝒟</sub>, m<sub>𝒟</sub> | hidden layers and width of each discriminator |
| `features` | dim. of **z** | dimension of the latent noise `z` (and output dimension of the latent encoder) |
| `activation` |  | `relu`, `elu`, `gelu` or `tanh` |
| `k1_over_k2` | (u<sub>𝒟</sub>, u<sub>𝒢/ℰ</sub>) | discriminator updates per generator update, u<sub>𝒟</sub> / u<sub>𝒢/ℰ</sub> (see [Training](Training.md#one-epoch)) |
| `lam` | λ | the latent term is `(1 - lam) * log_q` (`lam` is λ) |
| `beta` |  | weight of the reconstruction term (not the gate β<sub>t</sub>, which is set in `config.TASKS`) |
| `dropout` | d<sub>r</sub> | dropout inside the residual blocks of the generator |
| `num_layers_shared`, `neurons_shared` | N<sub>bb</sub>, m<sub>bb</sub> | blocks and width of the shared backbone (multi-task model only) |
| `seed` (optional) |  | overrides `SEED` for this model |

### Values of the three models

| | `tfz` | `dcn` | `dcn_tfz` |
|---|---|---|---|
| `lr_G`, `lr_T` | -3, -2 | -4, -3 | -3, -3 |
| `alpha_mean` | 0.1 | 0.1 | 0.01 (DCN), 0.1 (T<sub>fz</sub>) |
| `tau` | 0.01 | 0.1 | 0.01, 0.01 |
| `num_queries` | 2 | 1 | 1 |
| `batch_size` | 64 | 128 | 128 |
| `num_layers_trunk`, `neurons_trunk` | 2, 128 | 1, 256 | 2, 64 |
| `num_layers_D`, `neurons_D` | 1, 128 | 2, 128 | 1, 64 |
| `num_layers_E`, `neurons_E` | 1, 64 | 1, 64 | 2, 32 |
| `num_layers_T`, `neurons_T` | 4, 64 | 4, 64 | 2, 256 |
| `features` | 256 | 256 | 256 |
| `activation` | relu | relu | relu |
| `k1_over_k2` | 2.0 | 0.5 | 0.5 |
| `lam`, `beta`, `dropout` | 1.5, 1.0, 0.0 | 1.5, 1.0, 0.0 | 1.5, 1.0, 0.0 |
| `num_layers_shared`, `neurons_shared` | | | 2, 256 |

The values are those of the tables of the hyperparameter configurations in the Supplementary Material of the paper; each file is the most frequent configuration over the outer folds of the nested cross-validation (see [Training](Training.md#hyperparameters)).
