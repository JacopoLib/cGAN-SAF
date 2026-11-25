# Quick start

This page trains a model, writes its inference bundle and predicts a blend. [Installation](Installation.md) comes first.

## 1. Check the pipeline with a very short run

A full training can take a long time (up to 1000 epochs per model). To check that everything works, set `MAX_EPOCHS = 3` in `cgan/config.py`
and run one model:

```bash
python train_final.py --models tfz
```

On a laptop CPU, a run of a few epochs takes about a minute, including the loading of the data. The weights of such a run are not meaningful: set `MAX_EPOCHS` back to 1000.

## 2. Final training

```bash
python train_final.py                    # tfz, dcn and dcn_tfz, one after the other
python train_final.py --models dcn_tfz   # a single model
```

Each model is trained for at most 1000 epochs and stops after 100 epochs without improvement of the CRPS on the training data (see
[Training](Training.md#checkpoint-and-stopping)). The results are in `results/final/<model>/`:

| File | Content |
|---|---|
| `best_global.pt` | weights of the best epoch |
| `bundle_<task>.pt` | inference bundle of each task |
| `history.csv` | CRPS, RMSE and losses of every epoch |
| `config.csv` | hyperparameters used |
| `predictions_<task>.csv` | predictive mean, standard deviation and 95% interval next to the measured values, on the training data |

The folder `results/` is in `.gitignore`.

## 3. Predict

Pure compounds:

```bash
python predict.py --bundle results/final/tfz/bundle_tfz.pt --smiles CCCCCCCCCC "CC(C)CCCCC(C)C"
```

Blends (one row per component, volume fractions):

```
id,smiles,fraction
blend_1,CCCCCCCCCC,0.7
blend_1,Cc1ccccc1,0.3
```

```bash
python predict.py --bundle results/final/dcn_tfz/bundle_dcn.pt --input blends.csv --out predictions.csv
```

The output has one row per fuel with `mean`, `std`, `q2.5` and `q97.5` (see [Prediction](Prediction.md#output)). To use the trained weights of the paper, download the
bundles from Zenodo ([10.5281/zenodo.24087390](https://doi.org/10.5281/zenodo.24087390)) and pass their path to `--bundle`.

## 4. Change something

| I want to | See |
|---|---|
| use Mol2Vec instead of Mordred | [Descriptors](Descriptors.md#mol2vec) |
| change a hyperparameter | [Configuration reference](Configuration-reference.md) |
| add measurements | [Input data](Input-data.md#adding-data) |
| call the model from Python | [Prediction](Prediction.md#use-from-python) |
