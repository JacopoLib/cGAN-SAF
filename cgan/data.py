"""Property databases (DCN, freezing point) and the model-ready `Dataset`.

Every sample is a composition {SMILES: volume fraction}: a pure compound has a single component with fraction 1.
The species of the requested tasks form one vocabulary, on which descriptors are standardized.
"""

import ast
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import config
from .descriptors import clean_smiles, species_descriptors


def _read_list(path, encoding="utf-8"):
    """Read a Python-style list of SMILES written to a text file."""
    with open(path, encoding=encoding) as f:
        return [clean_smiles(s.strip("'\"")) for s in ast.literal_eval(f.read())]


def _read_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _blend_rows(folder, ids_file, values_file, vi_file):
    """Blend samples: one row per blend with a measured value, composition from the volume-fraction matrix."""
    components = _read_list(os.path.join(folder, "blends_components.txt"))
    vi = pd.read_csv(os.path.join(folder, vi_file), header=None).to_numpy()
    vi = vi / vi.sum(axis=1, keepdims=True)
    values = pd.read_csv(os.path.join(folder, values_file), header=None).to_numpy()[:, 0]
    ids = _read_lines(os.path.join(folder, ids_file))
    rows = []
    for i, (name, value) in enumerate(zip(ids, values)):
        if not np.isnan(value):
            rows.append((name, value, {components[k]: vi[i, k] for k in range(len(components)) if vi[i, k] > 0}))
    return rows


def load_dcn():
    """Derived cetane number: [(fuel id, DCN, composition)] for pure compounds (up to 9 values each) and blends."""
    folder = os.path.join(config.DATA_DIR, "dcn")
    pure = pd.read_csv(os.path.join(folder, "pure_species.csv"))
    values = np.load(os.path.join(folder, "pure_DCN.npy"))
    rows = []
    for name, smiles, row in zip(pure["name"], pure["smiles"], values):
        rows += [(name, v, {clean_smiles(smiles): 1.0}) for v in row if not np.isnan(v)]
    return rows + _blend_rows(folder, "blends_ids.txt", "blends_DCN.csv", "blends_vi.csv")


def load_tfz():
    """Freezing point [K]: [(fuel id, T_fz, composition)] for pure compounds and blends."""
    folder = os.path.join(config.DATA_DIR, "tfz")
    smiles = _read_list(os.path.join(folder, "pure_smiles.txt"), encoding="utf-16")
    names = _read_lines(os.path.join(folder, "pure_names.txt"))
    values = pd.read_csv(os.path.join(folder, "pure_Tfz.csv"), header=None).to_numpy()[:, 0]
    rows = [(n, v, {s: 1.0}) for n, s, v in zip(names, smiles, values) if not np.isnan(v)]
    return rows + _blend_rows(folder, "blends_ids.txt", "blends_Tfz.csv", "blends_vi.csv")


LOADERS = {"dcn": load_dcn, "tfz": load_tfz}
def species_vocabulary(tasks):
    """Ordered list of the distinct SMILES appearing in the databases of `tasks`."""
    return list(dict.fromkeys(smi for task in tasks for _, _, comp in LOADERS[task]() for smi in comp))


@dataclass
class Dataset:
    tasks: tuple  # task names, in output-column order
    fuels: (
        np.ndarray
    )  # (N,) fuel / blend identifiers (replicates of a fuel stay on the same side of the validation split)
    y: np.ndarray  # (N, n_tasks) z-scored targets, NaN where the task has no label
    task_id: np.ndarray  # (N,) task of each sample
    comp_idx: torch.Tensor  # (N, L) species index of each mixture component (-1 = padding)
    comp_frac: torch.Tensor  # (N, L) volume fraction of each component
    species: list  # species vocabulary (SMILES)
    Z_species: torch.Tensor  # (n_species, D) z-scored species descriptors
    desc_mean: torch.Tensor
    desc_std: torch.Tensor
    y_mean: np.ndarray  # (n_tasks,) target statistics, to go back to physical units
    y_std: np.ndarray
    blend_gate: list  # per task, see config.TASKS
    descriptor: str  # "mordred" or "mol2vec"
    descriptor_info: dict  # Mordred columns and medians, to describe new species at prediction time

    @property
    def n_tasks(self):
        return len(self.tasks)

    @property
    def latent_dim(self):
        return self.Z_species.shape[1]

    def denormalize(self, y_norm, task):
        return y_norm * self.y_std[task] + self.y_mean[task]


def build_dataset(tasks, descriptor=config.DESCRIPTOR):
    """Assemble the dataset of the requested tasks, e.g. ("tfz",) or ("dcn", "tfz")."""
    tasks = tuple(tasks)
    species = species_vocabulary(tasks)
    index = {s: i for i, s in enumerate(species)}

    Z, descriptor_info = species_descriptors(species, descriptor, selection=config.descriptor_selection(tasks))
    Z = torch.tensor(Z, dtype=torch.float32)
    desc_mean, desc_std = Z.mean(dim=0), Z.std(dim=0)
    desc_std[desc_std < 1e-8] = 1.0  # columns constant over the training species stay at zero
    Z = (Z - desc_mean) / desc_std

    fuels, y, task_id, comps = [], [], [], []
    for t, task in enumerate(tasks):
        rows = LOADERS[task]()
        fuels += [name for name, _, _ in rows]
        y += [[v if k == t else np.nan for k in range(len(tasks))] for _, v, _ in rows]
        task_id += [t] * len(rows)
        comps += [sorted((index[s], f) for s, f in comp.items()) for _, _, comp in rows]
    y = np.asarray(y, dtype=np.float32)
    y_mean, y_std = np.nanmean(y, axis=0), np.nanstd(y, axis=0)

    width = max(len(c) for c in comps)
    comp_idx = np.full((len(comps), width), -1, dtype=np.int64)
    comp_frac = np.zeros((len(comps), width), dtype=np.float32)
    for i, comp in enumerate(comps):
        comp_idx[i, : len(comp)] = [k for k, _ in comp]
        comp_frac[i, : len(comp)] = [f for _, f in comp]

    return Dataset(
        tasks=tasks,
        fuels=np.asarray([str(f) for f in fuels]),
        y=(y - y_mean) / y_std,
        task_id=np.asarray(task_id, dtype=np.int64),
        comp_idx=torch.from_numpy(comp_idx),
        comp_frac=torch.from_numpy(comp_frac),
        species=species,
        Z_species=Z,
        desc_mean=desc_mean,
        desc_std=desc_std,
        y_mean=y_mean,
        y_std=y_std,
        blend_gate=[config.TASKS[t]["blend_gate"] for t in tasks],
        descriptor=descriptor,
        descriptor_info=descriptor_info,
    )
