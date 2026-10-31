# Installation

## Python

Tested with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Each script needs only part of the requirements:

| Script | Packages |
|---|---|
| `rf_models.py` | numpy, pandas, scikit-learn, matplotlib |
| `bn_profiles.py` | + pgmpy, networkx |
| `gnn_embedding.py` | + torch, torch_geometric |
| `dl_model.py` | + tensorflow |

### PyTorch

A CPU build of PyTorch is enough. If `pip install -r requirements.txt` fails on `torch`, install it
first using the selector at <https://pytorch.org/get-started/locally/>, then rerun the command.

## R (optional: depth harmonisation only)

The harmonised datasets are already in `data/processed/`. R is only needed to regenerate them from
raw WoSIS layers. It needs the `ithir` package, which is distributed from GitHub rather than CRAN.
Follow the instructions at <https://github.com/brendo1001/ithir>.

## Tested environment

| Component | Version |
|---|---|
| OS | Linux (Ubuntu 24.04) |
| Python | 3.13 |
| scikit-learn / pandas / numpy | 1.6.1 / 2.2.3 / 2.2.6 |
| pgmpy | 0.1.26 |
| torch / torch_geometric | 2.7.1 / 2.8.1 |
| tensorflow | 2.21.0 |
| Hardware | 2 CPU cores, no GPU |

Approximate run times on this hardware: `rf_models.py` under 1 minute (3–4 minutes with `--cv`),
`dl_model.py`, `gnn_embedding.py` and `bn_profiles.py` under 1 minute each. `--compare` adds about 1 minute.
