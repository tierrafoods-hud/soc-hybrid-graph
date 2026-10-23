# User guide

How to apply the workflow to another region or dataset.

## Inputs

The Random Forest script and the graph notebooks expect one CSV per target property in
`data/processed/`, with one row per harmonised depth layer per profile and the columns listed in
[data/README.md](../data/README.md). The minimum is:

- `profile_id`, `depth_category` (`0_5`, `5_15`, `15_30`), `latitude`, `longitude`;
- the target (`orgc` or `tceq`, g/kg);
- covariates: any of `clay`, `elcosp`, `phaq`, `sand`, `silt`, `slopemean`, `cfr`, `temperature`,
  `precipitation`, `soil_type`, `bedrock`, `zone_number`, `landcover`.

Columns that are missing are skipped automatically. For stock (t/ha) tasks, `cfr` (coarse
fragments, % volume) is required.

## Configuration (`src/rf_models.py`)

| Setting | Where | Default |
|---|---|---|
| Input folder | `DATA_DIR` | `data/processed` |
| File name pattern | `load_and_preprocess` | `Mexico_standardized_cfvo_<target>.csv` |
| Numeric / categorical covariates | `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES` | see script |
| Depth layer thickness for stocks | `DEPTH_THICKNESS_CM` | 5, 10, 15 cm |
| Test fraction, seed | `TEST_SIZE`, `SEED` | 0.2, 42 |
| Split | `--split grouped` or `--split random` | `grouped` |

For a new region, put the files in `data/processed/` (or change `DATA_DIR` and the file name in
`load_and_preprocess`) and run `python src/rf_models.py`. The trained models are not transferable
between regions. Retrain and re-evaluate them on representative local data.

## Graph notebooks

`notebooks/dataset.py` defines the shared input table. Change the file names there for a new
dataset. Main parameters in `01_gnn_soil_embedding.ipynb`:

| Parameter | Meaning | Default |
|---|---|---|
| `NEIGHBOR_RADIUS_KM` | edge threshold between profiles | 10 km |
| `hidden_channels` / output size | GCN layer sizes | 128 / 64 |
| `EPOCHS`, `LEARNING_RATE`, `TRIPLET_MARGIN` | training settings | 50, 0.01, 1.0 |
| `n_clusters` | number of soil archetypes | 8 |

The notebook also prints a silhouette analysis to help choose `n_clusters` for a new region.

## Interpreting outputs

- `rf_metrics.csv`: test-set metrics per task. `profiles_in_both_sets` must be 0 for the
  grouped split.
- `rf_<task>_predicted_vs_actual.pdf`: observed vs predicted with the 1:1 line.
- `cluster_statistics.csv`: mean/sd of numeric variables and mode of categorical variables per cluster.
- The Bayesian network results are probabilistic associations under the assumed graph structure,
  not causal estimates.
