"""Paths and global settings."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DESCRIPTOR_DIR = os.path.join(DATA_DIR, "descriptors")
CONFIG_DIR = os.path.join(ROOT, "configs")
MOL2VEC_MODEL = os.environ.get("MOL2VEC_MODEL", os.path.join(DESCRIPTOR_DIR, "model_300dim.pkl"))

# Species descriptors fed to the models: "mordred" (default, computed and selected by cgan.descriptors) or
# "mol2vec" (needs gensim, mol2vec and the pre-trained Mol2Vec model). Can be overridden with --descriptor.
DESCRIPTOR = "mordred"

# Task order of the multi-task model and per-task settings.
# blend_gate (beta_t in the paper): whether the attention correction is multiplied by an is-blend indicator, so that
# it vanishes for pure compounds (gated, beta_t = 1) or not (non-gated, beta_t = 0).
TASKS = {
    "dcn": {"blend_gate": False},
    "tfz": {"blend_gate": True},
}

# The three models of the paper: name -> tasks. Their hyperparameters are in configs/<name>.csv.
MODELS = {
    "tfz": ("tfz",),
    "dcn": ("dcn",),
    "dcn_tfz": ("dcn", "tfz"),
}

# Selection of the Mordred columns of each model: "fixed" = list in data/descriptors/mordred_columns.txt, "filter" =
# species-dependent filter (constant, sparse and correlated columns removed) applied to the species of the model.
DESCRIPTOR_SELECTION = {"tfz": "fixed", "dcn": "fixed", "dcn_tfz": "filter"}


def descriptor_selection(tasks):
    """Mordred column selection of the model whose tasks are `tasks`."""
    for name, model_tasks in MODELS.items():
        if tuple(tasks) == model_tasks:
            return DESCRIPTOR_SELECTION[name]
    return "fixed"


N_AUX = 2  # reserved input columns (T, has_T); always zero for the properties considered here
SEED = 42

MAX_EPOCHS = 1000  # epoch budget of the final training
PATIENCE = 100  # early stopping: epochs without improvement of the CRPS on the training data
N_MC_SAMPLES = 1000  # Monte Carlo samples of the predictive distribution
