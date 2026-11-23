# Prediction

`predict.py` returns the predictive distribution of a property for pure compounds or blends, from an inference bundle written by
`train_final.py` (see [Training](Training.md#outputs)). The trained weights of the paper are archived on Zenodo
([10.5281/zenodo.23946612](https://doi.org/10.5281/zenodo.23946612)).

## Inference bundles

A bundle holds one task of one model. The single-task models give `bundle_tfz.pt` and `bundle_dcn.pt`; the multi-task model gives two bundles
that share the same network: `bundle_dcn.pt` and `bundle_tfz.pt` of `results/final/dcn_tfz/`. Choose the bundle of the property you want.

> **Security note.** The bundles are PyTorch files loaded with `torch.load(..., weights_only=False)`, which can execute arbitrary code if
> a file has been tampered with. Load only bundles that you trained yourself or that come from the Zenodo record above.

## Command line

Pure compounds:

```bash
python predict.py --bundle results/final/tfz/bundle_tfz.pt --smiles CCCCCCCCCC "CC(C)CCCCC(C)C"
```

Blends, from a CSV file with one row per component:

```bash
python predict.py --bundle results/final/dcn_tfz/bundle_dcn.pt --input blends.csv --out predictions.csv
```

```
id,smiles,fraction
blend_1,CCCCCCCCCC,0.7
blend_1,Cc1ccccc1,0.3
```

| Option | Default | Meaning |
|---|---|---|
| `--bundle` | required | `bundle_<task>.pt` |
| `--smiles` | | pure compounds, one or more SMILES |
| `--input` | | CSV with columns `id`, `smiles`, `fraction` (exactly one of `--smiles`, `--input`) |
| `--samples` | 1000 | Monte Carlo samples per fuel |
| `--out` | print to screen | output CSV |

### Input

- `fraction` is the **volume fraction** of the component. It is normalised to sum to 1 within each `id`, so any positive weights work.
- List each species once per `id`: if the same SMILES appears twice in an `id`, only the last row is kept.
- With `--smiles` the identifier of each fuel is its SMILES.
- A molecule that is in the training vocabulary may be written with **any valid SMILES**: it is canonicalised with RDKit and mapped to the
  spelling of the training vocabulary. If two spellings of the same molecule appear in one blend, their fractions are added. An invalid SMILES raises an error.
- A molecule that is not in the training vocabulary gets its descriptors computed on the fly (see [Descriptors](Descriptors.md#new-species)); this needs
  RDKit and mordred (or, for Mol2Vec bundles, gensim, mol2vec and the pre-trained model).

### Output

One row per fuel:

| Column | Content |
|---|---|
| `id` | identifier of the fuel |
| `mean`, `std` | mean and standard deviation of the Monte Carlo samples |
| `q2.5`, `q97.5` | 2.5% and 97.5% quantiles: the central 95% interval |

The values are in physical units: kelvin for the freezing point, the cetane-number scale for DCN. Without `--out`, the screen shows first the task of the
bundle and the tasks the model was trained on (`Task: dcn (model trained on: dcn, tfz)`).

## What the output is

For each fuel the code draws `n_samples` latent vectors `z ~ N(0, I)`, runs the generator on `(x, z)` with the mixture representation `x` of the fuel, and
converts back to physical units with the `y_mean` and `y_std` of the bundle. The mean, the standard deviation and the quantiles summarise these samples:
they describe the distribution that the generator has learned, conditional on the fuel. The validation of this distribution (coverage of the
intervals, comparison with other methods) is in the paper.

- The latent vectors come from a generator seeded with `config.SEED` (42), so the output is reproducible for a given bundle and input.
- The noise of a fuel depends on its **position** in the input: the same molecule listed twice receives different latent vectors, and its means
  differ by the Monte Carlo error. Use more `--samples` to reduce it.
- `predict.py` does not check whether a new species is inside the range of the training descriptors.

## Use from Python

```python
from predict import Predictor

p = Predictor("results/final/tfz/bundle_tfz.pt")          # device="cpu" by default
samples = p.sample([{"CCCCCCCCCC": 0.7, "Cc1ccccc1": 0.3},
                    {"CCCCCCCCCC": 1.0}], n_samples=1000)  # (2, 1000), physical units
```

`Predictor.sample(compositions, n_samples, seed)` takes a list of `{SMILES: fraction}` dictionaries and returns the array of samples.

During training, `cgan.final_training.predict(model, rows)` gives the samples of all the tasks of the multi-task model together, drawn from the same latent
vector.
