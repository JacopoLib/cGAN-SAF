# Installation

## Requirements

- Python 3.12 (the version used for the results of the paper)
- the packages in `requirements.txt`:

| Package | Used for |
|---|---|
| `torch` | networks and training |
| `numpy`, `pandas`, `scipy` | data handling; `scipy` for the Spearman correlations of the descriptor filter |
| `scikit-learn` | listed in `requirements.txt`; not imported by the code |
| `rdkit` | SMILES parsing, canonical SMILES, 3D conformers |
| `mordredcommunity` | Mordred descriptors of species that are not in the cache |
| `gensim`, `mol2vec` | **only** for `--descriptor mol2vec` |

The pinned versions are those used to produce the results of the paper. The code does not need a GPU: it uses CUDA when
`torch.cuda.is_available()` and the CPU otherwise.

## Install from the repository

```bash
git clone https://github.com/JacopoLib/cGAN-SAF.git
cd cGAN-SAF
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The code is not packaged: `train_final.py` and `predict.py` are run from the root of the clone, and the package `cgan/` is
imported from there.

## Check the installation

```bash
python -c "import cgan.config as c; print(c.MODELS)"
python train_final.py --help
```

## Data location

`cgan/config.py` locates `data/` and `configs/` relative to the source file (`ROOT` is the folder that contains `cgan/`). Do not
move them away from the root of the clone. Training results are written to `results/final/` in the **current folder**, so run the
scripts from the root of the clone.

## Optional: Mol2Vec

`--descriptor mol2vec` needs `gensim`, `mol2vec` and the pre-trained model `model_300dim.pkl`. Download the model from
<https://github.com/samoturk/mol2vec/tree/master/examples/models> and place it in `data/descriptors/`, or set the environment variable
`MOL2VEC_MODEL` to its path. The file is not distributed with the repository (it is in `.gitignore`).

## Common issues

- **`FileNotFoundError` on `data/...` or `configs/...`**: the folders were moved, or the files are not in the clone.
- **`ModuleNotFoundError: No module named 'cgan'`**: run the scripts from the root of the clone.
- **Failure in `mordred` or `rdkit` for a new species**: the molecule could not be embedded in 3D or one descriptor failed; its
  descriptors become missing values, which are replaced by the medians stored in the bundle (see [Descriptors](Descriptors.md#new-species)).
- **`torch.load` and untrusted files**: see the note in [Prediction](Prediction.md#inference-bundles).
