# Uncertainty-aware SAF property prediction with a conditional GAN and attention pooling

PyTorch code and datasets accompanying

> J. Liberatori, S. Iavarone, R. Vicquelin, *Uncertainty-aware sustainable aviation fuel property prediction using a
> conditional generative adversarial network with attention pooling*, Energy and AI (submitted).

The repository contains the final training of the **three c-GAN models** of the paper, each on its whole property
database and with the hyperparameters selected by the nested k-fold cross-validation (Section 2.5):

| Model | Tasks | Hyperparameters | Gate β_t |
| --- | --- | --- | --- |
| `tfz` | single-task, freezing point | [`configs/tfz.csv`](configs/tfz.csv) (Table S4-1) | gated (β = 1) |
| `dcn` | single-task, derived cetane number | [`configs/dcn.csv`](configs/dcn.csv) (Table S4-1) | non-gated (β = 0) |
| `dcn_tfz` | multi-task, DCN + freezing point | [`configs/dcn_tfz.csv`](configs/dcn_tfz.csv) | DCN non-gated, T_fz gated |

All models use Mordred descriptors for the pure compounds, multi-query attention pooling with feature augmentation
for mixtures, the softmin-of-K reconstruction loss and a checkpoint on the CRPS. The trained weights are
archived on Zenodo ([10.5281/zenodo.23946612](https://doi.org/10.5281/zenodo.23946612)); `predict.py` uses them to predict the property of new pure compounds and blends.

## Layout

```
train_final.py     final training of the three models -> weights and inference bundles
predict.py         predictive distribution for pure compounds or blends from an inference bundle
configs/           hyperparameters of the three models
cgan/
  config.py        paths, descriptor flag, per-task gate, training settings
  descriptors.py   Mordred (computed from the SMILES; fixed list or filtered) and Mol2Vec species descriptors
  data.py          freezing-point and DCN databases -> Dataset
  models.py        mixture encoder (attention pooling), generator, discriminators, latent encoder
  trainer.py       adversarial and softmin-of-K losses, CRPS and other metrics, training loop
  final_training.py  training with checkpoint, Monte Carlo predictions, inference bundles
  utils.py         seeding, task-balanced batches, early stopping / checkpoints
data/              property databases and cached raw Mordred descriptors
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Final training

```bash
python train_final.py                  # tfz, dcn and dcn_tfz, one after the other
python train_final.py --models tfz     # one model only (tfz, dcn or dcn_tfz)
```

Each model is trained on the whole database(s) for at most 1000 epochs. After every epoch the CRPS is computed on the
training data: the weights of the epoch with the lowest CRPS are kept, and training stops after 100 epochs without
improvement. The learning rates are halved when the training losses stop decreasing for 10 epochs.
`results/final/<model>/` then contains

* `best_global.pt`: weights of the best epoch (all networks);
* `bundle_<task>.pt`: inference bundle of each task (one for the single-task models, two for `dcn_tfz`), with the
  weights, the hyperparameters, the species descriptors and the normalization statistics;
* `history.csv`, `config.csv` and `predictions_<task>.csv` (predictive mean, standard deviation and 95% interval
  next to the measured values, for the training data).

## Prediction

> **Note.** The inference bundles are PyTorch files loaded with `torch.load`, which can execute arbitrary code
> if a file has been tampered with. Only load bundles obtained from the Zenodo record above.

```bash
python predict.py --bundle results/final/tfz/bundle_tfz.pt --smiles CCCCCCCCCC "CC(C)CCCCC(C)C"
python predict.py --bundle results/final/dcn_tfz/bundle_dcn.pt --input blends.csv --out predictions.csv
```

`blends.csv` lists one component per row:

```
id,smiles,fraction
blend_1,CCCCCCCCCC,0.7
blend_1,Cc1ccccc1,0.3
```

The output gives, for each pure compound or blend, the mean, standard deviation and 2.5/97.5% quantiles of 1000
Monte Carlo samples of the predictive distribution. Species outside the training databases are described on the fly
with the same descriptors as in training (this needs RDKit and mordred).

## Species descriptors

The flag `DESCRIPTOR` in `cgan/config.py` (or `--descriptor` on `train_final.py`) chooses the molecular representation:

* `mordred` (default): Mordred 1D/2D/3D descriptors computed from the SMILES and standardized. The columns are selected
  per model (`DESCRIPTOR_SELECTION` in `cgan/config.py`):
  * `tfz` and `dcn`: the fixed list of 347 descriptors in `data/descriptors/mordred_columns.txt`;
  * `dcn_tfz`: the columns left by a filter (columns that are mostly zero or missing, near-constant, or
    Spearman-correlated above 0.95 with an earlier column) applied to the species of both databases (309 columns).

  The raw descriptors of all species are shipped in `data/descriptors/mordred_raw.csv.gz`; run

  ```bash
  python -m cgan.descriptors --kind mordred --recompute
  ```

  to recompute them from the SMILES (several minutes; the 3D descriptors depend on the RDKit conformer generator, so
  they change slightly with the RDKit version). The inference bundles store the selected columns, so `predict.py`
  describes new species in the same way as in training.
* `mol2vec`: 300-dimensional Mol2Vec embeddings. Download the pre-trained model `model_300dim.pkl` from
  <https://github.com/samoturk/mol2vec/tree/master/examples/models> and place it in `data/descriptors/` (or set the
  `MOL2VEC_MODEL` environment variable).

## Data

| Folder | Content |
| --- | --- |
| `data/tfz` | freezing points [K] of 893 pure hydrocarbon measurements (`pure_*`) and 261 blend measurements (`blends_*`: components, volume fractions, values): 251 binary and 10 ternary blends |
| `data/dcn` | derived cetane numbers of 254 pure hydrocarbons (386 measurements, `pure_*`) and 543 blend measurements (`blends_*`) |
| `data/descriptors` | cached raw Mordred descriptors of all species and the fixed list of 347 columns (`mordred_columns.txt`) |

## Citation

```bibtex
@article{Liberatori2026cGAN,
  title   = {Uncertainty-aware sustainable aviation fuel property prediction using a conditional generative adversarial network with attention pooling},
  author  = {Liberatori, Jacopo and Iavarone, Salvatore and Vicquelin, Ronan},
  journal = {Energy and AI (submitted for publication)},
  year    = {2026}
}
```

## License

Copyright (C) 2026 The Authors. The code is released under the [Prosperity Public License 3.0.0](LICENSE)
(non-commercial use). The property databases in `data/` are compiled from the literature (including the tables of
Yaws); see the paper for the sources.
