# soc-hybrid-graph

Code and data to reproduce the results of the paper:

> Papadakis, E., Mallik, G., Ibrahim, A. A., Baryannis, G., Altinalmazis-Kondylis, A.
> *Integrating machine learning with spatial graph neural networks and Bayesian networks for
> explainable soil organic carbon assessment from legacy soil profile data.*
> Submitted to *Computers & Geosciences*.

The framework combines:

1. **Depth harmonisation** of WoSIS soil profile layers to 0–5, 5–15 and 15–30 cm with
   mass-preserving equal-area quadratic splines (R, `ithir`).
2. **Random Forest** models for organic carbon (ORGC) and calcium carbonate equivalent (TCEQ),
   as concentrations (g/kg) and stocks (t/ha), evaluated with a **profile-grouped** train/test split.
3. An exploratory **deep learning** model for ORGC concentration.
4. A **graph convolutional network** trained with a self-supervised triplet loss on a 10 km
   proximity graph of profile layers (carbon withheld from the inputs), followed by K-Means
   clustering into soil archetypes.
5. A **Bayesian network** (pgmpy) that profiles the environmental conditions associated with
   high and low organic carbon.

| | |
|---|---|
| Developers | Emmanuel Papadakis, Gulger Mallik, Ahmed Aman Ibrahim (University of Huddersfield) |
| Contact | George Baryannis, g.bargiannis@hud.ac.uk |
| Languages | Python 3 (tested with 3.13); R with `ithir` (depth harmonisation only) |
| Hardware | Standard desktop or laptop CPU (tested on 2 cores). No GPU needed. |
| Licence | Code: MIT ([LICENSE](LICENSE)). Data: WoSIS terms, see [data/README.md](data/README.md) |

## Repository layout

```
data/processed/        harmonised WoSIS-Mexico datasets used in the paper (see data/README.md)
data/mexico.geojson    state boundaries used for the cluster map
src/dataset.py         shared input table for the graph-based components
src/rf_models.py       Random Forest models          -> Table 1, Figures 3-4
src/dl_model.py        exploratory deep learning     -> Figures 5-6
src/gnn_embedding.py   GCN embeddings + K-Means       -> Figure 7, Tables 2-3
src/bn_profiles.py     Bayesian network profiles     -> Table 4
r/ea_spline_generator.R  equal-area spline depth harmonisation (Section 3.2)
outputs/               reference results written by the scripts
docs/                  INSTALL.md, TUTORIAL.md, USER_GUIDE.md
```

## Quick start

```bash
git clone https://github.com/tierrafoods-hud/soc-hybrid-graph.git
cd soc-hybrid-graph
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/rf_models.py --cv --compare   # Table 1, Figures 3-4, model comparison
python src/dl_model.py                   # Figures 5-6
python src/gnn_embedding.py              # Figure 7, Tables 2-3
python src/bn_profiles.py                # Table 4
```

See [docs/TUTORIAL.md](docs/TUTORIAL.md) for a step-by-step walkthrough.

## Reproducing the paper

| Paper item | Command | Output |
|---|---|---|
| Table 1, Figures 3–4 | `python src/rf_models.py` | `outputs/rf/grouped/` |
| Model comparison (Section 3.3) | `python src/rf_models.py --compare` | `outputs/rf/grouped/model_comparison.csv` |
| Cross-validation and leakage comparison (Section 5.1) | `--cv`, and `--split random` | `outputs/rf/` |
| Figures 5–6 (deep learning) | `python src/dl_model.py` | `outputs/dl/` |
| Figure 7, Tables 2–3 (soil archetypes) | `python src/gnn_embedding.py` | `outputs/gnn/carbon_excluded/` |
| Table 4 (Bayesian network profiles) | `python src/bn_profiles.py` | `outputs/bn/` |
| Depth harmonisation (Section 3.2) | `Rscript r/ea_spline_generator.R <in.csv> <out_dir>` | `<property>_harmonized.csv` |

All random seeds are fixed (42). Repeated runs on the same software versions give identical
results. Different library versions or hardware can change the GNN and deep learning results slightly.

Reference Random Forest results (80/20 split grouped by `profile_id`, so no profile is in both
the training and test sets):

| Task | Unit | MSE | RMSE | MAE | R² |
|---|---|---|---|---|---|
| ORGC | g/kg | 156.87 | 12.52 | 7.10 | 0.38 |
| TCEQ | g/kg | 15294.82 | 123.67 | 59.91 | 0.46 |
| ORGC stock | t/ha | 95.17 | 9.76 | 6.60 | 0.47 |
| TCEQ stock | t/ha | 28891.41 | 169.97 | 78.00 | 0.46 |

`--split random` reproduces the original row-level split, in which depth layers of the same profile
can be in both sets. It is included only to show the optimistic bias of that set-up (R² 0.60–0.82).

## Related repositories

The development history (data selection, covariate extraction, exploratory notebooks and the
Streamlit prototype) is in the other repositories of the
[tierrafoods-hud](https://github.com/tierrafoods-hud) organisation:
`notebooks`, `application`, `causal_graph`, `knowledge_graph`, `nasa_landcover`, `grids_mexico`.
They are not needed to reproduce the results above.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the WoSIS dataset (see `data/README.md`).

## Funding

Innovate UK, project reference 10121892.
