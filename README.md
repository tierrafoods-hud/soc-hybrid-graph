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
3. A **graph convolutional network** trained with a self-supervised triplet loss on a 10 km
   proximity graph of soil profiles, followed by K-Means clustering into soil archetypes.
4. A **Bayesian network** (pgmpy) used to profile the environmental conditions associated with
   high and low organic carbon.

| | |
|---|---|
| Developers | Emmanuel Papadakis, Gulger Mallik, Ahmed Aman Ibrahim (University of Huddersfield) |
| Contact | George Baryannis, g.bargiannis@hud.ac.uk |
| Languages | Python 3.11–3.13; R ≥ 4.2 (depth harmonisation only) |
| Hardware | Standard desktop or laptop (tested on 2 CPU cores). No GPU needed. |
| Licence | Code: MIT ([LICENSE](LICENSE)). Data: WoSIS terms, see [data/README.md](data/README.md) |

## Repository layout

```
data/processed/      harmonised WoSIS-Mexico datasets used in the paper (see data/README.md)
src/rf_models.py     Random Forest models: Table 1, Figures 3-4
notebooks/
  dataset.py                          shared input table for the graph notebooks
  01_gnn_soil_embedding.ipynb         GCN embeddings + K-Means clusters: Figure 7, Tables 2-3
  02_bayesian_network_profiles.ipynb  Bayesian network profiles: Table 4
r/ea_spline_generator.R  equal-area spline depth harmonisation (Section 3.2)
outputs/             results written by the scripts (reference copies included)
docs/                INSTALL.md, TUTORIAL.md, USER_GUIDE.md
```

## Quick start

```bash
git clone https://github.com/tierrafoods-hud/soc-hybrid-graph.git
cd soc-hybrid-graph
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/rf_models.py --cv       # Table 1 and Figures 3-4, a few minutes on a laptop
```

The metrics are written to `outputs/rf/grouped/rf_metrics.csv` and the figures to
`outputs/rf/grouped/*.pdf`. Then open the notebooks in `notebooks/` with Jupyter to reproduce
the graph-based results. See [docs/TUTORIAL.md](docs/TUTORIAL.md) for a step-by-step walkthrough.

## Reproducing the paper

| Paper item | Command / file | Output |
|---|---|---|
| Table 1, Figures 3–4 (Random Forest) | `python src/rf_models.py` | `outputs/rf/grouped/` |
| Figure 7, Tables 2–3 (GNN soil archetypes) | `notebooks/01_gnn_soil_embedding.ipynb` | notebook outputs, `cluster_statistics.csv` |
| Table 4 (Bayesian network profiles) | `notebooks/02_bayesian_network_profiles.ipynb` | notebook outputs |
| Depth harmonisation (Section 3.2) | `Rscript r/ea_spline_generator.R <in.csv> <out_dir>` | `<feature>_harmonized.csv` |

Reference results with the default settings (seed 42, 80/20 split grouped by `profile_id`, so
no profile appears in both the training and the test sets):

| Task | Unit | RMSE | MAE | R² |
|---|---|---|---|---|
| ORGC | g/kg | 12.52 | 7.10 | 0.38 |
| TCEQ | g/kg | 123.67 | 59.91 | 0.46 |
| ORGC stock | t/ha | 9.76 | 6.60 | 0.47 |
| TCEQ stock | t/ha | 169.97 | 78.00 | 0.46 |

`python src/rf_models.py --split random` reproduces the original row-level split, in which
depth layers of the same profile can be in both sets. It is included only to show the
optimistic bias of that set-up and was not used for the reported results.

## Related repositories

The development history (data selection, covariate extraction, exploratory models and the
Streamlit prototype) is in the other repositories of the
[tierrafoods-hud](https://github.com/tierrafoods-hud) organisation:
`notebooks`, `application`, `causal_graph`, `knowledge_graph`, `nasa_landcover`, `grids_mexico`.
They are not needed to reproduce the results above.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite the WoSIS dataset (see `data/README.md`).

## Funding

Innovate UK, project reference 10121892.
