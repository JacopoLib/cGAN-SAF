# Notation

The documentation uses the symbols of the paper. This page lists them next to the names used in the code, so that a formula can be
followed down to the line that implements it.

## Data and mixture representation

| Symbol | Meaning | Code |
|---|---|---|
| s | number of samples of a database | `len(ds)` |
| n | number of tasks (properties) of a model | `ds.n_tasks`; `tasks` of `config.MODELS` |
| γ, γ<sup>(t)</sup> | measured property (task t), z-scored during training | `y` |
| **Γ** | vector of the targets of the n tasks of one sample (NaN where a task has no label) | `y` of shape `(B, n)` |
| **x** = [**Φ**; T, p] | input of the networks | `x` of `CGAN.encode`, shape `(B, 2L + 2)` |
| T, p | temperature and pressure (not used for DCN and T<sub>fz</sub>) | the `N_AUX = 2` reserved columns, always zero |
| **φ**<sub>i</sub> | standardised descriptor vector of component i of a mixture | row of `Z_species`, `L = ds.latent_dim` |
| N | number of components of a mixture | length of `comp_idx` (padded with -1) |
| v<sub>i</sub> | volume fraction of component i | `comp_frac` (renormalised to 1) |
| **Φ** | fuel features: [Σ<sub>i</sub> v<sub>i</sub>**φ**<sub>i</sub> ‖ **φ**′<sub>mix</sub>] | the pair `(mean_pool, blend_delta)` of `MixtureEncoder` |

## Attention pooling

| Symbol | Meaning | Code |
|---|---|---|
| Q | number of learnable queries | `num_queries` (`configs/<model>.csv`) |
| **q**<sub>A</sub>, **q**<sub>B</sub>, … | learnable queries | `MultiQueryPooling.queries`, shape `(Q, L)` |
| α<sub>A</sub>, α<sub>B</sub>, … | learnable fraction bias of each query | `MultiQueryPooling.frac_scale` (initialised at 0.5) |
| **w**<sup>A</sup> | attention weights of the components for query A | softmax of the scores in `MultiQueryPooling.forward` |
| p<sub>A</sub>, p<sub>B</sub>, … | learnable softmax weights of the queries | `softmax(query_w)` |
| **φ**′<sub>mix</sub> | pooled mixture representation | `attn_pool` |
| β<sub>t</sub> | gate of task t: 1 gated, 0 non-gated | `blend_gate` of `config.TASKS` |

## Networks and latent variables

| Symbol | Meaning | Code |
|---|---|---|
| f<sub>θ</sub> | generator (single task); f<sub>Θ</sub><sup>(t)</sup> is the complete mapping of task t | `Generator`, head `BridgeNet`; attribute `net_D` of `CGAN` |
| θ<sub>bb</sub> | shared backbone (multi-task) | `SharedTrunk` |
| Θ | {θ<sub>bb</sub>, θ<sub>1</sub>, …, θ<sub>n</sub>}: backbone and generator heads | parameters of `net_D` |
| D<sub>ψ</sub>, D<sub>ψ<sub>t</sub></sub> | discriminator (of task t) | `Critic`; list `net_Ts` |
| q<sub>η</sub> | latent-space encoder | `net_E` |
| **z** | latent vector, **z** ~ N(**0**, **I**), dimension 256 | `features` (`n_f` in the code) |
| K = 32 | latent samples per data point in training and validation | `K_SAMPLES` in `cgan/trainer.py` |
| N<sub>MC</sub> = 1000 | Monte Carlo samples of the predictive distribution | `N_MC_SAMPLES` in `cgan/config.py` |

## Losses

| Symbol | Meaning | Code |
|---|---|---|
| 𝓛<sub>𝒟</sub> | discriminator loss | `CGAN.critic_loss` |
| 𝓛<sub>𝒢</sub> | generator loss | `CGAN.generator_loss` |
| 𝓛<sub>recon</sub><sup>(t)</sup> | softmin-of-K reconstruction penalty of task t | inside `generator_loss` |
| λ = 1.5 | latent-term parameter; the term is (1 − λ) times the log-density `log_q` | `lam` |
| α<sub>t</sub> | weight of the mean-anchor term of task t | `alpha_mean` or `alpha_mean_<t>` |
| τ<sub>t</sub> | temperature of the softmin of task t | `tau` or `tau_<t>` |
| CRPS | continuous ranked probability score, the checkpoint metric M<sub>ck</sub> | `crps_ensemble`, metric `crps` |

> The column `beta` of `configs/<model>.csv` is **not** the gate β<sub>t</sub>: it is the weight of the whole reconstruction term, 1 in all the final
> models. The gate is set per task in `config.TASKS`.

## Hyperparameters

| Symbol | Meaning | Column of `configs/<model>.csv` |
|---|---|---|
| b | batch size | `batch_size` |
| l<sub>r,𝒢/ℰ</sub>, l<sub>r,𝒟</sub> | learning rates of the generator/encoder block and of the discriminators (the file stores log10) | `lr_G`, `lr_T` |
| d<sub>r</sub> | dropout rate of the residual blocks of the generator networks | `dropout` |
| N<sub>𝒢,1</sub>, m<sub>𝒢,1</sub> | residual blocks and width of stage 1 of the generator (head) | `num_layers_trunk`, `neurons_trunk` |
| N<sub>𝒢,2</sub>, m<sub>𝒢,2</sub> | residual blocks and width of stage 2 of the generator (head) | `num_layers_D`, `neurons_D` |
| N<sub>𝒟</sub>, m<sub>𝒟</sub> | layers and width of each discriminator | `num_layers_T`, `neurons_T` |
| N<sub>ℰ</sub>, m<sub>ℰ</sub> | residual blocks and width of the encoder | `num_layers_E`, `neurons_E` |
| N<sub>bb</sub>, m<sub>bb</sub> | residual blocks and width of the shared backbone | `num_layers_shared`, `neurons_shared` |
| (u<sub>𝒟</sub>, u<sub>𝒢/ℰ</sub>) | discriminator and generator/encoder updates per mini-batch | `k1_over_k2` = u<sub>𝒟</sub> / u<sub>𝒢/ℰ</sub> |

## Training and validation

| Symbol | Meaning | Code |
|---|---|---|
| N<sub>max</sub> = 1000 | maximum number of epochs | `MAX_EPOCHS` |
| P<sub>max</sub> = 100 | early-stopping patience | `PATIENCE` |
| M<sub>ck</sub> | checkpoint metric: the CRPS, averaged over the tasks | `crps` of `CGAN.evaluate` |
| k<sub>outer</sub> = 5, k<sub>inner</sub> = 3 | folds of the nested cross-validation used to select the hyperparameters | not in the repository |
| N<sub>RS</sub> = 10 N<sub>hp</sub> = 160 | random-search configurations of each outer fold | not in the repository |
