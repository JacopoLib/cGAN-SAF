"""Predictive distribution of a property for pure compounds or blends, from an inference bundle of train_final.py.

    python predict.py --bundle results/final/tfz/bundle_tfz.pt --smiles CCCCCCCCCC Cc1ccccc1
    python predict.py --bundle results/final/dcn_tfz/bundle_dcn.pt --input blends.csv --out predictions.csv

The input CSV has one row per component, with columns `id`, `smiles` and `fraction` (volume fractions; they are
normalized within each id). A molecule of the training databases may be written with any valid SMILES; species
that are not in the training databases get their descriptors computed on the fly (Mordred: RDKit + mordred; Mol2Vec: gensim + mol2vec and the pre-trained model).
"""

import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from cgan import config
from cgan.descriptors import clean_smiles, new_species_descriptors
from cgan.models import Generator, MixtureEncoder
from cgan.trainer import HyperParameters


class Predictor:
    """Generator and mixture encoder of one task, rebuilt from an inference bundle."""

    def __init__(self, bundle_path, device="cpu"):
        b = torch.load(bundle_path, map_location=device, weights_only=False)
        self.b, self.device, self.task = b, device, b["task_id"]
        self.hp = HyperParameters.from_config(
            b["config"], SimpleNamespace(n_tasks=len(b["task_names"]), blend_gate=b["blend_gate"])
        )
        L, T = b["Z_species"].shape[1], len(b["task_names"])
        self.mix_enc = MixtureEncoder(L, self.hp.num_queries, self.hp.blend_gate).to(device)
        self.net_D = Generator(
            T,
            self.hp.features,
            2 * L,
            self.hp.head_widths,
            self.hp.head_layers,
            self.hp.activation,
            self.hp.dropout,
            self.hp.shared,
        ).to(device)
        self.mix_enc.load_state_dict(b["mix_enc_state"])
        self.net_D.load_state_dict(b["net_D_state"])
        self.mix_enc.eval()
        self.net_D.eval()
        self.index = {s: i for i, s in enumerate(b["species"])}
        self.canonical = {}  # canonical SMILES -> the SMILES of the same molecule in the training vocabulary
        for s in b["species"]:
            self.canonical.setdefault(Chem.CanonSmiles(s), s)

    def _repository_smiles(self, smiles):
        """The training-vocabulary spelling of a known molecule; unknown molecules keep the SMILES they came with."""
        smiles = clean_smiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles!r}")
        return self.canonical.get(Chem.MolToSmiles(mol), smiles)

    def _species_matrix(self, smiles):
        """Standardized descriptors of the training species plus those of the new species in `smiles`."""
        b = self.b
        new = [s for s in dict.fromkeys(smiles) if s not in self.index]
        Z = b["Z_species"]
        index = dict(self.index)
        if new:
            raw = torch.tensor(new_species_descriptors(new, b["descriptor"], b["descriptor_info"]), dtype=torch.float32)
            Z = torch.cat([Z, (raw - b["desc_mean"]) / b["desc_std"]])
            index.update({s: len(self.index) + i for i, s in enumerate(new)})
        return Z.to(self.device), index

    def _canonicalize(self, composition):
        """{SMILES: fraction} with every known molecule written as in the training vocabulary (fractions of SMILES that
        turn out to be the same molecule are added)."""
        out = {}
        for smiles, fraction in composition.items():
            key = self._repository_smiles(smiles)
            out[key] = out.get(key, 0.0) + float(fraction)
        return out

    @torch.no_grad()
    def sample(self, compositions, n_samples=config.N_MC_SAMPLES, seed=config.SEED):
        """Monte Carlo samples (n_fuels, n_samples) in physical units; compositions: list of {SMILES: fraction}."""
        compositions = [self._canonicalize(c) for c in compositions]
        Z, index = self._species_matrix([s for c in compositions for s in c])
        width = max(len(c) for c in compositions)
        comp_idx = torch.full((len(compositions), width), -1, dtype=torch.long)
        comp_frac = torch.zeros((len(compositions), width))
        for i, c in enumerate(compositions):
            comp_idx[i, : len(c)] = torch.tensor([index[s] for s in c])
            comp_frac[i, : len(c)] = torch.tensor(list(c.values()))
        task_id = torch.full((len(compositions),), self.task, dtype=torch.long)
        mean_pool, delta = self.mix_enc(comp_idx.to(self.device), comp_frac.to(self.device), Z, task_id.to(self.device))
        x = torch.cat([mean_pool, delta, torch.zeros(len(compositions), config.N_AUX, device=self.device)], dim=1)

        g = torch.Generator(device=self.device).manual_seed(seed)
        out = np.empty((len(compositions), n_samples), dtype=np.float32)
        y_mean, y_std = float(self.b["y_mean"][self.task]), float(self.b["y_std"][self.task])
        for n in range(n_samples):
            z = torch.randn(len(compositions), self.hp.features, generator=g, device=self.device)
            out[:, n] = (self.net_D(x, z)[:, self.task] * y_std + y_mean).cpu().numpy()
        return out


def read_compositions(args):
    if args.smiles:
        return list(args.smiles), [{s: 1.0} for s in args.smiles]
    df = pd.read_csv(args.input)
    ids = list(dict.fromkeys(df["id"]))
    return ids, [dict(zip(g["smiles"], g["fraction"])) for _, g in df.groupby("id", sort=False)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="bundle_<task>.pt written by train_final.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", nargs="+", help="pure compounds")
    source.add_argument("--input", help="CSV with columns id, smiles, fraction (one row per component)")
    parser.add_argument("--samples", type=int, default=config.N_MC_SAMPLES, help="Monte Carlo samples per fuel")
    parser.add_argument("--out", help="output CSV (default: print to screen)")
    args = parser.parse_args()

    predictor = Predictor(args.bundle)
    ids, compositions = read_compositions(args)
    s = predictor.sample(compositions, args.samples)
    result = pd.DataFrame(
        {
            "id": ids,
            "mean": s.mean(axis=1),
            "std": s.std(axis=1),
            "q2.5": np.quantile(s, 0.025, axis=1),
            "q97.5": np.quantile(s, 0.975, axis=1),
        }
    )
    print(f"Task: {predictor.b['task_name']} (model trained on: {', '.join(predictor.b['task_names'])})")
    if args.out:
        result.to_csv(args.out, index=False)
        print("Written:", args.out)
    else:
        print(result.to_string(index=False))
