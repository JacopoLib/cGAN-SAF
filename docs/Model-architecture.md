# Model architecture

> **Implementation map**
>
> | Concept | Code |
> |---|---|
> | Mixture encoder (mean pooling + attention pooling, gate β<sub>t</sub>) | `MixtureEncoder`, `MultiQueryPooling` (`cgan/models.py`) |
> | Generator (one head per task, optional shared trunk) | `Generator`, `BridgeNet`, `SharedTrunk` |
> | Discriminator (critic) of one task | `Critic` |
> | Latent-space encoder | `MLP` with `use_residual=True`, attribute `net_E` of `CGAN` |
> | Assembly, input of the networks | `CGAN`, `CGAN.encode` (`cgan/trainer.py`) |
> | Hyperparameters | `HyperParameters` (`cgan/trainer.py`), `configs/<model>.csv` |

The model has four networks: the **mixture encoder**, the **generator** `net_D`, one **critic** per task `net_Ts` and a **latent encoder**
`net_E`. Only the mixture encoder and the generator are needed to predict. Below, `L` is the number of descriptors per species
(`ds.latent_dim`), `B` the batch size, `T` the number of tasks and `n_f = features` the dimension of the latent noise.

## Mixture encoder

Input: for each sample, the species indices of its components (padded with -1), their volume fractions, the standardised descriptor
matrix `Z_species` and the task of the sample.

1. The fractions of the real components are renormalised to sum to 1 (padding gets fraction 0).
2. **Mean pooling**: `mean_pool = sum_j f_j Z_j`, the fraction-weighted mean of the component descriptors. Shape `(B, L)`.
3. **Attention pooling** (`MultiQueryPooling`), with `K = num_queries` learnable queries `q_k` of dimension `L`. For query `k`
   and component `j`,

   ```
   score_kj = q_k . Z_j / sqrt(L)  +  s_k * log(max(f_j, 1e-12))
   ```

   where `s_k` is a learnable scale per query (initialised at 0.5), so that the volume fraction biases the attention. Padding is
   masked with `-inf`. A softmax over the components gives the weights, and the pooled vector of query `k` is the weighted sum of
   the `Z_j`. The `K` pooled vectors are combined with a softmax over learnable weights `query_w` (uniform at initialisation) and summed:
   `attn_pool`, shape `(B, L)`.
4. **Gate β<sub>t</sub>** (`blend_gate`, `config.TASKS`): `blend_delta = attn_pool * scale`, with `scale = is_blend` for a gated task
   and `scale = 1` for a non-gated task. `is_blend` is 1 when the sample has more than one component and 0 for a pure compound. For a gated task the
   attention correction is therefore exactly zero for pure compounds.

The output is the pair `(mean_pool, blend_delta)`. `CGAN.encode` concatenates it with `N_AUX = 2` columns of zeros (reserved for a
temperature and a "has temperature" flag, always zero for the properties considered here) into the input `x` of shape
`(B, 2L + 2)`.

The gate of each task is set by `config.TASKS`: `dcn` is non-gated, `tfz` is gated. In the multi-task model the gate is applied
per sample, according to the task of the sample.

## Generator

Input `(x, z)`, with `z ~ N(0, I)` of dimension `n_f`. Only the first `2L` columns of `x` are used (the reserved columns are ignored). The
generator returns one value per task, in z-scored units.

Each task has its own head, `BridgeNet`:

```
[h, z] -> Linear -> LayerNorm -> act                       (gateway, width W1)
       -> n1 residual blocks of width W1 -> Linear         (phase 1)
       -> Linear(W1 -> W2) -> LayerNorm -> act             (bridge)
       -> n2 residual blocks of width W2 -> Linear(W2 -> 1)  (phase 2, output)
```

A residual block is `x + fc2(dropout(act(fc1(x))))`. In `configs/<model>.csv`: `W1 = neurons_trunk`, `n1 = num_layers_trunk`, `W2 = neurons_D`,
`n2 = num_layers_D`, `dropout` is applied inside the residual blocks.

- **Single-task models** (`tfz`, `dcn`): `h = x[:, :2L]`, the head input has dimension `2L + n_f`.
- **Multi-task model** (`dcn_tfz`): `h` is the output of a `SharedTrunk`: `Linear(2L -> neurons_shared) -> LayerNorm -> act`,
  `num_layers_shared` residual blocks, `Linear`. Every head takes `[h, z]`.

The same `z` is given to all the heads for a sample, so that the samples of the different tasks of the same draw are generated from one latent vector.

## Critics

One critic per task (`Critic`): an `MLP` that scores the pair `(x, y)` (input dimension `2L + 2 + 1`). It has an input layer, `num_layers_T`
hidden layers of width `neurons_T` (each Linear + LayerNorm + activation) and a linear output, with **spectral normalisation** on every
linear layer. By the convention of the loss (see [Training](Training.md#losses)) the critic output is **low for real pairs and high for generated pairs**.

## Latent encoder

`net_E` is a residual `MLP` with `num_layers_E` blocks of width `neurons_E`. It maps `[x, y_gen]` (dimension `2L + 2 + T`, where `y_gen` is
the generated value, set to zero for the tasks without a label) to a vector of dimension `n_f`. It is used only in the training loss
(latent-space regularisation) and is not needed to predict.

## Activation functions

`activation` in the configuration is one of `relu`, `elu`, `gelu`, `tanh`. An unknown name gives an identity in the `MLP` (critics, latent
encoder) and a ReLU in the generator (`BridgeNet`, `SharedTrunk`), so use one of the four names.

## Weights initialisation

Every `Linear` layer of an `MLP`, `SharedTrunk` and `BridgeNet` is initialised with Xavier-uniform weights and zero biases. The attention
queries are drawn from `N(0, 1)`.

## Sizes of the three final models

| | `tfz` | `dcn` | `dcn_tfz` |
|---|---|---|---|
| descriptors `L` | 347 | 347 | 309 |
| generator input `x` (used columns) | 694 | 694 | 618 |
| attention queries | 2 | 1 | 2 |
| shared trunk (width x blocks) | none | none | 256 x 2 |
| head, phase 1 (width x blocks) | 128 x 2 | 256 x 1 | 64 x 2 |
| head, phase 2 (width x blocks) | 128 x 1 | 128 x 2 | 128 x 2 |
| critic (width x hidden layers) | 64 x 4 | 64 x 4 | 128 x 1 (per task) |
| latent encoder (width x blocks) | 64 x 1 | 64 x 1 | 64 x 1 |
| latent dimension `n_f` | 256 | 256 | 256 |
| activation, dropout | ReLU, 0 | ReLU, 0 | ReLU, 0 |
