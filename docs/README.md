# c-GAN for SAF property prediction: documentation

This repository contains a **conditional generative adversarial network (c-GAN) with attention pooling** that predicts the
*distribution* of a property of a pure hydrocarbon or of a blend, from the molecular structure (SMILES) and the volume fractions of
its components. It is used for two properties of sustainable aviation fuels (SAFs): the **derived cetane number** (DCN) and the
**freezing point** (T<sub>fz</sub>).

> Liberatori J., Iavarone S., Vicquelin R., *Uncertainty-aware sustainable aviation fuel property prediction using a
> conditional generative adversarial network with attention pooling*, Energy and AI (submitted for publication).

## In one paragraph

Every pure compound is described by a vector of Mordred descriptors (or by a Mol2Vec embedding). A blend is described by the
descriptors of its components, pooled in two ways: the volume-fraction-weighted mean, and a correction obtained by multi-query
attention pooling. The generator takes this mixture representation and a latent noise vector `z` and returns one sample of the
property; repeating the draw with different `z` gives the predictive distribution. The generator is trained against one
discriminator (critic) per property, with a latent-space regularisation and a softmin-of-K reconstruction loss. The final models of the
paper are three: freezing point alone, DCN alone, and DCN + freezing point in one multi-task network.

## What the repository contains

| Included | Not included |
|---|---|
| final training of the three models of the paper on the whole databases, with the hyperparameters selected in the paper | the nested k-fold cross-validation used to select the hyperparameters (described in Section 2.5 of the paper) |
| prediction of new pure compounds and blends from an inference bundle | the trained weights (archived on Zenodo, [10.5281/zenodo.23946612](https://doi.org/10.5281/zenodo.23946612)) |
| the two property databases and the cached Mordred descriptors | |

## Pages

### Using the code

| Page | Content |
|---|---|
| [Installation](Installation.md) | requirements, environment, optional packages |
| [Quick start](Quick-start.md) | train a model, predict a blend, outputs of a run |
| [Prediction](Prediction.md) | `predict.py`: input formats, new species, what the output means |
| [Configuration reference](Configuration-reference.md) | `cgan/config.py` and every column of `configs/<model>.csv` |
| [Input data](Input-data.md) | the two databases: files, formats, units, counts |
| [API reference](API-reference.md) | modules, classes and functions |
| [Development](Development.md) | repository layout, reproducibility, how to extend |

### What the code does

| Page | Content |
|---|---|
| [Descriptors](Descriptors.md) | Mordred (fixed list or filter) and Mol2Vec, standardisation, new species |
| [Model architecture](Model-architecture.md) | mixture encoder, generator, critics, latent encoder |
| [Training](Training.md) | losses, update schedule, batching, checkpoint on the CRPS, bundles |

Each of the last three pages starts with a table that maps the concepts to the code. The pages describe the code as it is
implemented; for the motivation and the validation of the method see the paper.

## The three models

| Model | Tasks | Hyperparameters | Gate β<sub>t</sub> | Mordred columns |
|---|---|---|---|---|
| `tfz` | freezing point | `configs/tfz.csv` | gated (1) | fixed list (347) |
| `dcn` | derived cetane number | `configs/dcn.csv` | non-gated (0) | fixed list (347) |
| `dcn_tfz` | DCN and freezing point (multi-task) | `configs/dcn_tfz.csv` | DCN non-gated, T<sub>fz</sub> gated | filter (309) |

The gate β<sub>t</sub> is explained in [Model architecture](Model-architecture.md#mixture-encoder).

## Status

Python 3.12 with the package versions in `requirements.txt` (the versions used to produce the results of the paper). Runs on CPU or on a
CUDA GPU.
