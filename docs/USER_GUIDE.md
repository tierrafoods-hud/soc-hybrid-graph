# User guide

How to apply the workflow to another region or dataset.

## Inputs

The scripts expect one CSV per target property in `data/processed/`, with one row per harmonised
depth layer per profile and the columns listed in [data/README.md](../data/README.md). The minimum is:

- `profile_id`, `depth_category` (`0_5`, `5_15`, `15_30`), `latitude`, `longitude`;
- the target (`orgc` or `tceq`, g/kg);
- covariates: any of `clay`, `elcosp`, `phaq`, `sand`, `silt`, `slopemean`, `cfr`, `temperature`,
  `precipitation`, `soil_type`, `bedrock`, `zone_number`, `landcover`.

Missing covariate columns are skipped by `rf_models.py`. For stock (t/ha) tasks, `cfr` (coarse
fragments, % volume) is required. The graph scripts (`gnn_embedding.py`, `bn_profiles.py`) need all
columns used in `src/dataset.py`.

## Configuration

| Script | Setting | Default |
|---|---|---|
| `rf_models.py` | `DATA_DIR`, file pattern in `load_and_preprocess` | `data/processed/Mexico_standardized_cfvo_<target>.csv` |
| | `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES` | see script |
| | `DEPTH_THICKNESS_CM` (stock calculation) | 5, 10, 15 cm |
| | `TEST_SIZE`, `SEED`, `--split` | 0.2, 42, `grouped` |
| `dl_model.py` | `DATA_FILE`, `FEATURES` | single-depth ORGC file; clay, pH, sand, silt |
| `gnn_embedding.py` | `NEIGHBOR_RADIUS_KM` | 10 km |
| | `HIDDEN_DIM`, `EMBEDDING_DIM` | 128, 64 |
| | `EPOCHS`, `LEARNING_RATE`, `TRIPLET_MARGIN` | 50, 0.01, 1.0 |
| | `N_CLUSTERS`, `--include-carbon` | 8, off |
| `bn_profiles.py` | `EDGES` (network structure), `CATEGORICAL`, `SAND_BINS` | see script |
| `dataset.py` | input files for the graph components | ORGC and TCEQ files |

For a new region, put the files in `data/processed/` (or change the paths above) and rerun the
scripts. The trained models are not transferable between regions. Retrain and re-evaluate them on
representative local data. `gnn_embedding.py` reports the silhouette score for the chosen number
of clusters in `summary.json`.

## Interpreting outputs

- `outputs/rf/<split>/rf_metrics.csv`: test-set metrics per task. `profiles_in_both_sets` must be
  0 for the grouped split.
- `outputs/rf/<split>/model_comparison.csv`: R² and RMSE of all candidate models.
- `outputs/dl/dl_metrics.json`: deep learning test metrics, epochs trained and best validation MAE.
- `outputs/gnn/carbon_excluded/cluster_numeric_stats.csv` / `cluster_categorical_modes.csv`:
  Tables 2–3. `cluster_assignments.csv` gives the cluster of every node.
- `outputs/bn/bn_orgc_profiles.csv`: P(state | orgc = high/low) for every variable. These are
  probabilistic associations under the assumed network structure, not causal estimates.
