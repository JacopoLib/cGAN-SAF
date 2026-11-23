# Development

## Repository layout

```
cGAN-SAF/
├── cgan/              package: configuration, data, descriptors, networks, training
├── configs/           hyperparameters of the three models
├── data/              property databases and cached raw Mordred descriptors
├── docs/              documentation
├── train_final.py
├── predict.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Workflow

- Commit messages: a short imperative summary, then a paragraph on the reason when it is not obvious.
- Do not commit run outputs (`results/`, `*.pt`), the Mol2Vec model or `.DS_Store`; they are in `.gitignore`.

## Checking a change

After modifying the data loading, the networks, the losses or the prediction, run a short training (set `MAX_EPOCHS = 3` in `cgan/config.py`
for the check) of a single-task and of the multi-task model, and check that the run completes, that `history.csv` is sensible and that
`predict.py` works on the new bundle. With the same seed and the same environment, a change that does not alter the method must give the same `history.csv`.

## Reproducibility

- `config.SEED` seeds the training and the Monte Carlo draws of `predict.py`; a `seed` column in `configs/<model>.csv` overrides it for a model.
- Record the commit hash, the descriptor kind and the versions of PyTorch and RDKit with the run. The 3D descriptors depend on the RDKit version.

## Extending the code

- **A new model** (a different combination of the existing tasks): add it to `config.MODELS` and `config.DESCRIPTOR_SELECTION`, and create `configs/<name>.csv`.
- **A new property**: add a loader in `cgan/data.py` and register it in `LOADERS`, add the task to `config.TASKS` (with its gate), and add the model.
- **A new descriptor**: add it to `species_descriptors` and `new_species_descriptors` in `cgan/descriptors.py` and to the `choices` of `--descriptor`.

## License

Prosperity Public License 3.0.0 (non-commercial). See `LICENSE`.

## Maintaining the documentation

The documentation lives in `docs/` (Markdown, rendered in the repository browser) and is versioned together with the code. A change that alters the behaviour or the parameters
should update the matching page.
