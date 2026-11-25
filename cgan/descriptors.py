"""Molecular descriptors of the pure species: Mordred (default) or Mol2Vec.

Mordred descriptors are computed from the SMILES (1D/2D/3D, one MMFF-optimised conformer per molecule), cached in
`data/descriptors/mordred_raw.csv.gz`. The models use the fixed list of columns in `data/descriptors/mordred_columns.txt`. Run

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

from . import config


def clean_smiles(smiles):
    """Strip all whitespace (including non-breaking spaces) from a SMILES string."""
    return re.sub(r"\s+", "", str(smiles).replace(" ", ""))


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


def mordred_descriptors(smiles, cache_dir=config.DESCRIPTOR_DIR, recompute=False):
    """Mordred descriptor matrix (n_species, n_descriptors) for `smiles`, and the information that maps new species onto
    the same descriptors (column names and medians used to impute missing values), see `mordred_transform`.

    The columns are those listed in `data/descriptors/mordred_columns.txt`; missing values are replaced by the column
    medians over `smiles`."""
    raw = mordred_raw(smiles, os.path.join(cache_dir, "mordred_raw.csv.gz"), recompute)
    columns = mordred_columns(os.path.join(cache_dir, "mordred_columns.txt"))
    X = raw.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    medians = np.nanmedian(X, axis=0)
    rows, cols = np.nonzero(np.isnan(X))
    X[rows, cols] = medians[cols]
    return X, {"columns": columns, "medians": medians}


def mordred_transform(smiles, info, cache_dir=config.DESCRIPTOR_DIR):
    """Mordred descriptors of (possibly new) species, restricted to the columns of the fixed list."""
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


def species_descriptors(smiles, kind=config.DESCRIPTOR, recompute=False):
    """Descriptor matrix (n_species, n_descriptors) of the given species, with `kind` in {"mordred", "mol2vec"}, and
    the information needed to describe new species in the same way (see `new_species_descriptors`)."""
    smiles = [clean_smiles(s) for s in smiles]
    if kind == "mordred":
        return mordred_descriptors(smiles, recompute=recompute)
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
    X, _ = species_descriptors(vocabulary, kind, args.recompute)
    print(f"{kind}: {X.shape[0]} species, {X.shape[1]} descriptors")
