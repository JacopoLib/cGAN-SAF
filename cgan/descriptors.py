"""Molecular descriptors of the pure species: Mordred (default) or Mol2Vec.

Mordred descriptors are computed from the SMILES (1D/2D/3D, one MMFF-optimised conformer per molecule), cached in
`data/descriptors/mordred_raw.csv.gz`. Two selections of the Mordred columns are available (`DESCRIPTOR_SELECTION` in
`config.py`): the fixed list in `data/descriptors/mordred_columns.txt` ("fixed") and the species-dependent filter of
the original MATLAB routine ("filter"). Run

    python -m cgan.descriptors --kind mordred

to build the cache (add --recompute to ignore the shipped one). The 3D descriptors depend on the RDKit conformer
generator, so values recomputed with another RDKit version differ slightly from the shipped cache.
Mol2Vec needs the `mol2vec` and `gensim` packages and the pre-trained model `model_300dim.pkl` (see README).
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from . import config

ZERO_FRAC_MAX = 0.95  # filter: drop columns that are zero in more than 95% of the species
NAN_FRAC_MAX = 0.30  # filter: drop columns with more than 30% missing values
STD_TOL = 1e-8  # filter: drop (near-)constant columns
CORR_THR = 0.95  # filter: |Spearman| above which the later column of a pair is dropped


def clean_smiles(smiles):
    """Strip all whitespace (including non-breaking spaces) from a SMILES string."""
    return re.sub(r"\s+", "", str(smiles).replace(" ", ""))


def filter_descriptors(X):
    """Port of the MATLAB `filter_descriptors`. Returns the filtered, median-imputed matrix and the kept-column mask."""
    X = np.asarray(X, dtype=float)
    keep = np.ones(X.shape[1], dtype=bool)
    keep &= ~((X == 0).mean(axis=0) > ZERO_FRAC_MAX)
    keep &= ~(np.isnan(X).mean(axis=0) > NAN_FRAC_MAX)
    idx = np.flatnonzero(keep)
    Xk = X[:, idx].copy()

    col_median = np.nanmedian(Xk, axis=0)
    rows, cols = np.nonzero(np.isnan(Xk))
    Xk[rows, cols] = col_median[cols]

    varying = Xk.std(axis=0, ddof=1) > STD_TOL
    idx, Xk = idx[varying], Xk[:, varying]

    keep_corr = np.ones(Xk.shape[1], dtype=bool)
    if Xk.shape[1] >= 2:
        corr = np.abs(spearmanr(Xk).statistic)
        pairs = np.argwhere(np.triu(corr, 1) > CORR_THR)
        pairs = pairs[np.lexsort((pairs[:, 0], pairs[:, 1]))]  # column-major order, as in MATLAB `find`
        for i, j in pairs:
            if keep_corr[i] and keep_corr[j]:
                keep_corr[j] = False
    idx, Xk = idx[keep_corr], Xk[:, keep_corr]

    mask = np.zeros(X.shape[1], dtype=bool)
    mask[idx] = True
    return Xk, mask


def _mol_3d(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = True
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        if AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94") is not None:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    return mol


def compute_mordred(smiles):
    """Raw Mordred descriptors (all 1D/2D/3D) of a list of SMILES; failed molecules give NaN rows."""
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=False)
    rows = []
    for i, smi in enumerate(smiles, 1):
        print(f"Mordred {i}/{len(smiles)}: {smi}", flush=True)
        mol = _mol_3d(smi)
        try:
            rows.append({str(k): v for k, v in calc(mol).asdict().items()} if mol is not None else {})
        except Exception:
            rows.append({})
    df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
    df.insert(0, "SMILES", list(smiles))
    return df


def mordred_raw(smiles, cache_path, recompute=False, update_cache=True):
    """Raw Mordred table for `smiles`; only the species missing from the cache are computed (all if `recompute`)."""
    smiles = [clean_smiles(s) for s in smiles]
    cached = pd.read_csv(cache_path) if os.path.exists(cache_path) and not recompute else None
    missing = smiles if cached is None else [s for s in dict.fromkeys(smiles) if s not in set(cached["SMILES"])]
    if missing:
        new = compute_mordred(list(dict.fromkeys(missing)))
        cached = new if cached is None else pd.concat([cached, new], ignore_index=True)
        if update_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cached.to_csv(cache_path, index=False)  # gzip-compressed when the path ends with .gz
    return cached.drop_duplicates("SMILES").set_index("SMILES").loc[smiles]


def mordred_columns(path=os.path.join(config.DESCRIPTOR_DIR, "mordred_columns.txt")):
    """Names of the Mordred descriptors used by the models (fixed list, shipped in data/descriptors)."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def mordred_descriptors(smiles, selection="fixed", cache_dir=config.DESCRIPTOR_DIR, recompute=False):
    """Mordred descriptor matrix (n_species, n_descriptors) for `smiles`, and the information that maps new species onto
    the same descriptors (column names and medians used to impute missing values), see `mordred_transform`.

    selection="fixed": the columns listed in `data/descriptors/mordred_columns.txt`; missing values are replaced by the
    column medians over `smiles`.
    selection="filter": the columns retained by `filter_descriptors` applied to `smiles`, i.e. to the species of the
    databases of the requested tasks."""
    raw = mordred_raw(smiles, os.path.join(cache_dir, "mordred_raw.csv.gz"), recompute)
    if selection == "filter":
        X, mask = filter_descriptors(raw.to_numpy(dtype=float))
        columns = list(raw.columns[mask])
        return X, {"columns": columns, "medians": np.nanmedian(raw[columns].to_numpy(dtype=float), axis=0)}
    if selection != "fixed":
        raise ValueError(f"unknown descriptor selection {selection!r}")
    columns = mordred_columns(os.path.join(cache_dir, "mordred_columns.txt"))
    X = raw.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    medians = np.nanmedian(X, axis=0)
    rows, cols = np.nonzero(np.isnan(X))
    X[rows, cols] = medians[cols]
    return X, {"columns": columns, "medians": medians}


def mordred_transform(smiles, info, cache_dir=config.DESCRIPTOR_DIR):
    """Mordred descriptors of (possibly new) species, restricted to the columns retained by `mordred_descriptors`."""
    raw = mordred_raw(smiles, os.path.join(cache_dir, "mordred_raw.csv.gz"), update_cache=False)
    X = raw.reindex(columns=info["columns"]).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    rows, cols = np.nonzero(np.isnan(X))
    X[rows, cols] = np.asarray(info["medians"])[cols]
    return X


def mol2vec_descriptors(smiles, model_path=config.MOL2VEC_MODEL):
    """300-dimensional Mol2Vec embeddings: sum of the vectors of the Morgan substructure 'words' of each molecule.

    Needs the `mol2vec` and `gensim` packages and the pre-trained model `model_300dim.pkl`."""
    from gensim.models import word2vec
    from mol2vec.features import mol2alt_sentence
    from rdkit import Chem

    model = word2vec.Word2Vec.load(model_path)
    known = set(model.wv.key_to_index)
    vectors = []
    for smi in smiles:
        sentence = mol2alt_sentence(Chem.MolFromSmiles(smi), 1)
        vectors.append(sum(model.wv.get_vector(w if w in known else "UNK") for w in sentence))
    return np.asarray(vectors)


def species_descriptors(smiles, kind=config.DESCRIPTOR, recompute=False, selection="fixed"):
    """Descriptor matrix (n_species, n_descriptors) of the given species, with `kind` in {"mordred", "mol2vec"}, and
    the information needed to describe new species in the same way (see `new_species_descriptors`)."""
    smiles = [clean_smiles(s) for s in smiles]
    if kind == "mordred":
        return mordred_descriptors(smiles, selection, recompute=recompute)
    if kind == "mol2vec":
        return mol2vec_descriptors(smiles), {}
    raise ValueError(f"unknown descriptor kind {kind!r}")


def new_species_descriptors(smiles, kind, info):
    """Raw (not standardized) descriptors of species outside the training vocabulary."""
    smiles = [clean_smiles(s) for s in smiles]
    return mordred_transform(smiles, info) if kind == "mordred" else mol2vec_descriptors(smiles)


if __name__ == "__main__":
    from .data import LOADERS, species_vocabulary

    parser = argparse.ArgumentParser(description="Build the species descriptor cache.")
    parser.add_argument("--tasks", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    parser.add_argument("--kind", default=config.DESCRIPTOR, choices=["mordred", "mol2vec"])
    parser.add_argument(
        "--recompute", action="store_true", help="ignore the cache and recompute every Mordred descriptor"
    )
    args = parser.parse_args()
    kind = args.kind
    vocabulary = species_vocabulary(args.tasks)
    X, _ = species_descriptors(vocabulary, kind, args.recompute, config.descriptor_selection(args.tasks))
    print(f"{kind}: {X.shape[0]} species, {X.shape[1]} descriptors")
