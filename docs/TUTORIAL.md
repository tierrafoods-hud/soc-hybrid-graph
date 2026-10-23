# Tutorial: reproducing the results step by step

This walkthrough reproduces the paper's results from the harmonised datasets in `data/processed/`,
then shows how to regenerate those datasets from WoSIS. Install the requirements first
([INSTALL.md](INSTALL.md)).

## 1. Look at the input data

```python
import pandas as pd
df = pd.read_csv("data/processed/Mexico_standardized_cfvo_orgc.csv")
print(df.shape)                      # (10016, 21)
print(df["profile_id"].nunique())    # 3326 profiles
print(df["depth_category"].value_counts())
```

Each soil profile contributes up to three rows, one per harmonised layer (0–5, 5–15, 15–30 cm).
The column descriptions are in [data/README.md](../data/README.md).

## 2. Random Forest models (Table 1, Figures 3–4)

```bash
python src/rf_models.py --cv
```

For each of the four tasks (ORGC and TCEQ, in g/kg and t/ha) the script:

1. drops columns with more than 70 % missing values, then duplicate and incomplete rows;
2. for the t/ha tasks, converts concentration to stock per layer:
   `stock = C × BD × thickness × (1 − coarse fragments) × 0.1`, with bulk density BD from the
   Drew (1973) pedotransfer function (ORGC) or 1.3 g/cm³ (TCEQ);
3. splits the data 80/20 **by profile** (`GroupShuffleSplit`, seed 42), so all layers of a
   profile are either in the training or in the test set (`profiles_in_both_sets` = 0);
4. fits a pipeline of mean/mode imputation, standardisation, one-hot encoding and
   `RandomForestRegressor(n_estimators=100, random_state=42)`, fitted on the training set only;
5. reports MSE, RMSE, MAE and R² on the test set and saves a predicted-vs-actual plot.

With `--cv` it also reports 5-fold profile-grouped cross-validation (mean ± sd), which shows how
much the scores vary across different profile splits.

Outputs:

```
outputs/rf/grouped/rf_metrics.csv
outputs/rf/grouped/rf_<task>_predicted_vs_actual.pdf
```

Options:

```bash
python src/rf_models.py --tasks orgc_gkg orgc_tha   # run selected tasks only
python src/rf_models.py --split random              # original row-level split (comparison only)
```

## 3. Graph-based components (Figure 7, Tables 2–4)

```bash
cd notebooks
jupyter lab
```

Both notebooks load the same table through `notebooks/dataset.py`, which merges the ORGC and TCEQ
files (TCEQ is used as inorganic carbon, `inorgc`).

**`01_gnn_soil_embedding.ipynb`**

- builds a graph with one node per profile layer and an undirected edge between nodes whose
  Haversine distance is ≤ 10 km (`NEIGHBOR_RADIUS_KM`);
- uses node features that exclude `orgc` and `inorgc`;
- trains a two-layer GCN (128 → 64) with a triplet margin loss (margin 1.0) for 50 epochs with
  Adam (learning rate 0.01). Anchor/positive pairs are graph neighbours; negatives are
  non-neighbours;
- clusters the 64-dimensional embeddings with K-Means (k = 8, `random_state=42`), maps the
  clusters and writes per-cluster statistics to `cluster_statistics.csv` (Tables 2–3).

GCN training samples triplets at random and the original notebook does not fix the PyTorch seed,
so cluster numbering and exact statistics can differ slightly between runs. The saved notebook
outputs are the ones reported in the paper.

**`02_bayesian_network_profiles.ipynb`**

- discretises the continuous variables into low/medium/high classes by quantiles (e.g. pH ≤ 5.95 / ≥ 8.08);
- fits a pgmpy `BayesianNetwork` (maximum likelihood) on the assumed graph structure defined in the notebook (`edges`);
- queries the network with `VariableElimination`, conditioning on high and low `orgc`, to obtain
  the environmental profiles in Table 4.

## 4. Regenerating the harmonised datasets (optional)

1. Download the WoSIS snapshot (December 2023) from the
   [ISRIC Data Hub](https://data.isric.org/srv/eng/catalog.search#/metadata/e50f84e1-aa5b-49cb-bd6b-cd581232a2ec)
   and extract the profiles and layers for Mexico (`country_name == "Mexico"`), with columns
   `profile_id`, `upper_depth`, `lower_depth` and the soil properties.
2. Harmonise each property to 0–5, 5–15 and 15–30 cm:

   ```bash
   Rscript r/ea_spline_generator.R wosis_mexico_layers.csv outputs/harmonised
   ```

   This fits an equal-area quadratic spline per profile (`ithir::ea_spline`) and writes one
   `<property>_harmonized.csv` per property.
3. Join the climate, land-cover and ecoregion covariates by location
   (see [data/README.md](../data/README.md)).
