"""Final training of the c-GAN models of the paper on the whole property databases, with fixed hyperparameters.

    python train_final.py                       # the three models: tfz, dcn (single-task) and dcn_tfz (multi-task)
    python train_final.py --models tfz          # single-task freezing point only
    python train_final.py --models dcn_tfz      # multi-task model only

Each model reads its hyperparameters from configs/<model>.csv and writes to results/final/<model>/: the best weights
(best_global.pt), one inference bundle per task (bundle_<task>.pt, used by predict.py), the training history, the
hyperparameters and the predictions on the training data.
"""

import argparse
import os

from cgan import config, final_training, utils
from cgan.data import build_dataset

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--models", nargs="+", default=list(config.MODELS), choices=list(config.MODELS))
parser.add_argument("--descriptor", default=config.DESCRIPTOR, choices=["mordred", "mol2vec"])
parser.add_argument("--out", default=os.path.join("results", "final"))
args = parser.parse_args()

for name in args.models:
    tasks = config.MODELS[name]
    print(f"\n=== {name}: {'multi' if len(tasks) > 1 else 'single'}-task model ({', '.join(tasks)}) ===")
    cfg = final_training.load_config(name)
    utils.enable_determinism(config.SEED)
    ds = build_dataset(tasks, args.descriptor)
    print(f"{len(ds.fuels)} samples, {len(ds.species)} species, {ds.latent_dim} {args.descriptor} descriptors")
    out = os.path.join(args.out, name if args.descriptor == config.DESCRIPTOR else f"{name}_{args.descriptor}")
    final_training.run_final_training(ds, cfg, out)
