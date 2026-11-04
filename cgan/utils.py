"""Reproducibility helpers, balanced batching and checkpointing."""

import os
import random

import numpy as np
import torch
from torch.utils.data import Sampler


def enable_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FixedBatchSampler(Sampler):
    """Yields precomputed batches (lists of dataset positions), optionally in a shuffled order."""

    def __init__(self, batches, shuffle_batches=False, seed=0):
        self.batches = [list(map(int, b)) for b in batches]
        self.shuffle_batches = shuffle_batches
        self.seed = seed

    def __iter__(self):
        order = (
            np.random.RandomState(self.seed).permutation(len(self.batches))
            if self.shuffle_batches
            else range(len(self.batches))
        )
        for k in order:
            yield self.batches[k]

    def __len__(self):
        return len(self.batches)


def balanced_batches(task_of_row, batch_size, n_tasks, seed):
    """Batches with the same number of samples of every task. `task_of_row[i]` is the task of dataset position i."""
    rng = np.random.default_rng(seed)
    positions = []
    for t in range(n_tasks):
        pos = np.nonzero(task_of_row == t)[0]
        rng.shuffle(pos)
        positions.append(pos)
    per_task = max(1, batch_size // n_tasks)
    batches = []
    for b in range(min(len(p) // per_task for p in positions)):
        batch = np.concatenate([p[b * per_task : (b + 1) * per_task] for p in positions])
        rng.shuffle(batch)
        batches.append(batch)
    return batches


class Checkpointer:
    """Early stopping on a validation score (lower is better), optionally saving the best weights."""

    def __init__(self, patience, path=None):
        self.best = float("inf")
        self.bad_epochs = 0
        self.patience = int(patience)
        self.path = path
        if path is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def step(self, score, epoch, state_fn):
        """Returns (stop, improved). `state_fn()` builds the dict to save when the score improves."""
        improved = score < self.best
        if improved:
            self.best, self.bad_epochs = float(score), 0
            if self.path is not None:
                torch.save({"epoch": int(epoch), "best_val_score": float(score), **state_fn()}, self.path)
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience, improved
