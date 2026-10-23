# Installation

## Python (required)

Python 3.11, 3.12 or 3.13.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`src/rf_models.py` needs only `numpy`, `pandas`, `scikit-learn` and `matplotlib`. To run just the
Random Forest models on a machine where PyTorch cannot be installed:

```bash
pip install numpy==2.2.6 pandas==2.2.3 scikit-learn==1.6.1 matplotlib==3.10.0
```

### PyTorch and PyTorch Geometric

The GNN notebook needs `torch` and `torch_geometric`. A CPU build is enough. If
`pip install -r requirements.txt` fails on `torch`, install it first using the selector at
<https://pytorch.org/get-started/locally/>, then run `pip install torch_geometric`.

Check the installation:

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

## R (optional: depth harmonisation only)

The harmonised datasets are already in `data/processed/`. R is only needed to regenerate them from
raw WoSIS layers.

- R ≥ 4.2
- The `ithir` package, which is distributed from GitHub rather than CRAN. Follow the instructions at
  <https://github.com/brendo1001/ithir>.

## Tested environment

| Component | Version |
|---|---|
| OS | Linux (Ubuntu) |
| Python | 3.13 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| pgmpy | 0.1.26 |
| Hardware | 2 CPU cores, no GPU |

`src/rf_models.py --cv` runs in about 3–4 minutes on this hardware (under 1 minute without `--cv`).
