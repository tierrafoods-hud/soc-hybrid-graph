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

Each soil profile contributes up to three rows, one per harmonised layer (0–5, 5–15 and
15–30 cm). The column descriptions are in [data/README.md](../data/README.md).

## 2. Random Forest models (Table 1, Figures 3–4)

```bash
python src/rf_models.py --cv --compare
```

For each of the four tasks (ORGC and TCEQ, in g/kg and t/ha) the script:

1. drops columns with more than 70 % missing values, then duplicate rows and rows with missing values;
2. for the t/ha tasks, converts concentration to stock per layer:
   `stock = C × BD × thickness × (1 − coarse fragments) × 0.1`, with bulk density
   `BD = 1.62 − 0.06 × OM` (`OM = 1.724 × C`) for ORGC and 1.3 g/cm³ for TCEQ. The concentration
   is then removed from the inputs;
3. splits the data 80/20 **by profile** (`GroupShuffleSplit`, seed 42), so all layers of a
   profile are either in the training or in the test set (`profiles_in_both_sets` = 0);
4. fits a pipeline of standardisation, one-hot encoding and
   `RandomForestRegressor(n_estimators=100, random_state=42)`, fitted on the training set only;
5. reports MSE, RMSE, MAE and R² on the test set and saves a predicted-vs-actual plot.

`--cv` adds 5-fold profile-grouped cross-validation (mean ± sd). `--compare` fits Gradient
Boosting, linear regression, MLP and SVR models on the same split and writes
`model_comparison.csv`. `--split random` reproduces the original row-level split for comparison.

## 3. Exploratory deep learning model (Figures 5–6)

```bash
python src/dl_model.py
```

A feed-forward network (64–32–16 units, ReLU, batch normalisation, dropout) predicts ORGC (g/kg)
from clay, sand, silt and pH. The input is `Mexico_single_depth_orgc_merged.csv`, which has one
harmonised 0–30 cm layer per profile, so the random 80/20 split is also a profile-level split.
Training uses Adam with MSE loss, 20 % of the training data for validation and early stopping
(patience 20). Outputs: `outputs/dl/dl_metrics.json` and the two figures.

## 4. Graph neural network soil archetypes (Figure 7, Tables 2–3)

```bash
python src/gnn_embedding.py
```

- One node per profile layer (3,916 nodes, from `src/dataset.py`).
- Node features: standardised sand, electrical conductivity, slope, silt-clay, pH and rock
  fragments; one-hot land cover, soil type, ecoregion, bedrock and depth layer; latitude and
  longitude. **Organic and inorganic carbon are excluded.** `--include-carbon` adds them, as in the
  original development notebook, for comparison.
- Undirected edges between nodes within 10 km (Haversine). Layers of the same profile are always connected.
- Two-layer GCN (128 → 64, ReLU), triplet margin loss (margin 1.0), 50 epochs, Adam (lr 0.01).
  Positives are graph neighbours; negatives are random non-neighbours.
- K-Means (k = 8) on the embeddings. The script writes cluster statistics (Tables 2–3), cluster
  assignments, a summary and the map (Figure 7) to `outputs/gnn/carbon_excluded/`.

## 5. Bayesian network profiles (Table 4)

```bash
python src/bn_profiles.py
```

- Continuous variables are discretised into low/medium/high terciles (sand uses fixed classes
  < 20, 20–45 and ≥ 45 %). Depth layer, land cover, ecoregion, bedrock and soil type are categorical.
- A pgmpy `BayesianNetwork` with the assumed structure in `EDGES` is fitted by maximum likelihood.
- The network is queried with `VariableElimination`, conditioning on `orgc = high` and `orgc = low`.
  All conditional distributions are written to `outputs/bn/bn_orgc_profiles.csv`, and the most
  probable states are printed.

## 6. Regenerating the harmonised datasets (optional)

1. Download the WoSIS snapshot (December 2023) from the
   [ISRIC Data Hub](https://data.isric.org/srv/eng/catalog.search#/metadata/e50f84e1-aa5b-49cb-bd6b-cd581232a2ec)
   and extract the profiles and layers for Mexico, with columns `profile_id`, `upper_depth`,
   `lower_depth` and the soil properties.
2. Harmonise each property to 0–5, 5–15 and 15–30 cm:

   ```bash
   Rscript r/ea_spline_generator.R wosis_mexico_layers.csv outputs/harmonised
   ```

   This fits an equal-area quadratic spline per profile (`ithir::ea_spline`) and writes one
   `<property>_harmonized.csv` per property.
3. Attach the covariates by location. See [data/README.md](../data/README.md) for the sources and
   which extraction scripts are available.
