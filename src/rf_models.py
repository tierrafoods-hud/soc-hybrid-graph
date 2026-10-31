"""Random Forest models for soil organic carbon (Table 1, Figures 3-4).

Reproduces the supervised component of the paper from the harmonised
(0-5, 5-15, 15-30 cm) WoSIS-Mexico datasets. Preprocessing, features and
model settings follow the original development notebooks
(tierrafoods-hud/notebooks: soil-model-training-gkg/tha.ipynb).

The train/test split is grouped by ``profile_id`` so that all depth layers
of a soil profile fall entirely in either the training or the test set.
Use ``--split random`` to reproduce the original row-level split for
comparison only.

Usage
-----
    python src/rf_models.py                      # all four tasks, grouped split
    python src/rf_models.py --cv                 # + 5-fold profile-grouped cross-validation
    python src/rf_models.py --compare            # + comparison with GB, linear, MLP and SVR models
    python src/rf_models.py --split random       # original row-level split
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "rf"

SEED = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["clay", "elcosp", "phaq", "sand", "silt", "slopemean", "cfr",
                    "temperature", "precipitation"]
CATEGORICAL_FEATURES = ["depth_category", "soil_type", "bedrock", "zone_number", "landcover"]

# Thickness (cm) of each harmonised depth interval, used for stock calculation.
DEPTH_THICKNESS_CM = {"0_5": 5, "5_15": 10, "15_30": 15}
# Ordinal encoding of depth used as a feature in the stock models (as in the original notebook).
DEPTH_ORDINAL = {"0_5": 0.5, "5_15": 1.0, "15_30": 2.0}

TASKS = {
    # task name: (dataset target, unit, convert to stock?)
    "orgc_gkg": ("orgc", "g/kg", False),
    "tceq_gkg": ("tceq", "g/kg", False),
    "orgc_tha": ("orgc", "t/ha", True),
    "tceq_tha": ("tceq", "t/ha", True),
}


def load_and_preprocess(target: str) -> pd.DataFrame:
    """Load the harmonised dataset and apply the cleaning steps of the original notebooks."""
    df = pd.read_csv(DATA_DIR / f"Mexico_standardized_cfvo_{target}.csv")
    df = df.drop(columns=["country_name", "region", "continent", "ecoregion_type", "zone_name"],
                 errors="ignore")
    df = df.dropna(subset=[target])
    threshold = 0.7 * len(df)  # drop columns with more than 70 % missing values
    df = df.drop(columns=[c for c in df.columns if df[c].isna().sum() > threshold and c != target])
    df = df.drop_duplicates().dropna()
    df["zone_number"] = df["zone_number"].astype(str)
    return df.reset_index(drop=True)


def add_stock(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, str]:
    """Convert concentration (g/kg) to stock (t/ha) per depth layer.

    Stock = C * BD * thickness * (1 - coarse fragments) * 0.1, with bulk density
    from the Drew (1973) pedotransfer function when organic carbon is available,
    otherwise 1.3 g/cm3. The concentration column is removed from the inputs.
    """
    df = df.copy()
    if "orgc" in df.columns:
        bulk_density = 1.62 - 0.06 * (df["orgc"] * 1.724 / 10)
    else:
        bulk_density = 1.3
    thickness = df["depth_category"].map(DEPTH_THICKNESS_CM)
    stock_col = f"{target}_stock"
    df[stock_col] = df[target] * bulk_density * thickness * (1 - df["cfr"] / 100) * 0.1
    df["depth_category"] = df["depth_category"].map(DEPTH_ORDINAL)
    return df.drop(columns=[target]), stock_col


def split(df: pd.DataFrame, how: str):
    if how == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
        train_idx, test_idx = next(gss.split(df, groups=df["profile_id"]))
        return df.iloc[train_idx], df.iloc[test_idx]
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED)
    return train, test


def regressors() -> dict:
    """Candidate models compared during development (same settings as the original notebooks)."""
    return {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=SEED),
        "Linear Regression": LinearRegression(),
        "MLP Regressor": MLPRegressor(hidden_layer_sizes=(2, 2), max_iter=1000, random_state=SEED),
        "SVR": SVR(kernel="poly"),
    }


def build_model(numeric: list[str], categorical: list[str], regressor=None) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessor", pre),
                     ("regressor", regressor if regressor is not None
                      else RandomForestRegressor(n_estimators=100, random_state=SEED))])


def plot(y_true, y_pred, target: str, unit: str, stats: dict, path: Path) -> None:
    """Predicted-vs-observed scatter in the style of the paper's Figures 3-4."""
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context({"font.size": 15}):
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        fig, ax = plt.subplots(figsize=(7, 5.6))
        ax.scatter(y_true, y_pred, alpha=0.6, s=10, color="#4c72b0", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="crimson", lw=2)
        ax.text(0.02, 0.98, f"RMSE={stats['RMSE']:.3f}  MAE={stats['MAE']:.3f}  R2={stats['R2']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "grey"})
        ax.set_xlabel(f"Actual values ({unit})")
        ax.set_ylabel(f"Predicted values ({unit})")
        ax.set_title(f"Predicted vs Actual: Random Forest ({target})")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def cross_validate(df: pd.DataFrame, y_col: str, numeric: list[str], categorical: list[str],
                   folds: int = 5) -> dict:
    """Profile-grouped k-fold cross-validation (supplementary robustness check)."""
    scores = {"RMSE": [], "MAE": [], "R2": []}
    for tr, te in GroupKFold(n_splits=folds).split(df, groups=df["profile_id"]):
        train, test = df.iloc[tr], df.iloc[te]
        pred = build_model(numeric, categorical).fit(train[numeric + categorical], train[y_col]) \
            .predict(test[numeric + categorical])
        scores["RMSE"].append(float(np.sqrt(mean_squared_error(test[y_col], pred))))
        scores["MAE"].append(mean_absolute_error(test[y_col], pred))
        scores["R2"].append(r2_score(test[y_col], pred))
    return {f"cv_{k}_{f}": float(getattr(np, f)(v)) for k, v in scores.items() for f in ("mean", "std")}


def compare_models(name: str, how: str) -> list[dict]:
    """Test-set R2/RMSE of all candidate models on the same split (model selection check)."""
    target, unit, to_stock = TASKS[name]
    df = load_and_preprocess(target)
    y_col = target
    if to_stock:
        df, y_col = add_stock(df, target)
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    train, test = split(df, how)
    rows = [{"task": name, "model": "Baseline (training mean)", "RMSE": float(np.sqrt(mean_squared_error(
        test[y_col], np.full(len(test), train[y_col].mean())))), "R2": r2_score(
        test[y_col], np.full(len(test), train[y_col].mean()))}]
    for label, reg in regressors().items():
        pred = build_model(numeric, categorical, reg).fit(train[numeric + categorical], train[y_col]) \
            .predict(test[numeric + categorical])
        rows.append({"task": name, "model": label, "RMSE": float(np.sqrt(mean_squared_error(test[y_col], pred))),
                     "R2": r2_score(test[y_col], pred)})
    return rows


def run_task(name: str, how: str, cv: bool = False) -> dict:
    target, unit, to_stock = TASKS[name]
    df = load_and_preprocess(target)
    y_col = target
    if to_stock:
        df, y_col = add_stock(df, target)
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    train, test = split(df, how)
    shared = set(train["profile_id"]) & set(test["profile_id"])
    model = build_model(numeric, categorical).fit(train[numeric + categorical], train[y_col])
    pred = model.predict(test[numeric + categorical])

    mse = mean_squared_error(test[y_col], pred)
    res = {
        "task": name, "target": y_col, "unit": unit, "split": how,
        "n_rows": len(df), "n_profiles": int(df["profile_id"].nunique()),
        "n_train": len(train), "n_test": len(test),
        "profiles_in_both_sets": len(shared),
        "MSE": mse, "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(test[y_col], pred), "R2": r2_score(test[y_col], pred),
    }
    if cv and how == "grouped":
        res.update(cross_validate(df, y_col, numeric, categorical))
    out = OUT_DIR / how
    out.mkdir(parents=True, exist_ok=True)
    plot(test[y_col].to_numpy(), pred, y_col, unit, res, out / f"rf_{name}_predicted_vs_actual.pdf")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["grouped", "random"], default="grouped")
    parser.add_argument("--cv", action="store_true",
                        help="also report 5-fold profile-grouped cross-validation (grouped split only)")
    parser.add_argument("--compare", action="store_true",
                        help="also compare Random Forest with the other candidate models")
    parser.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    args = parser.parse_args()

    results = [run_task(t, args.split, args.cv) for t in args.tasks]
    table = pd.DataFrame(results)
    out = OUT_DIR / args.split
    table.to_csv(out / "rf_metrics.csv", index=False)
    (out / "rf_metrics.json").write_text(json.dumps(results, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[["task", "unit", "n_train", "n_test", "profiles_in_both_sets",
                     "MSE", "RMSE", "MAE", "R2"]].round(4).to_string(index=False))
        if args.cv and args.split == "grouped":
            print(table[["task"] + [c for c in table.columns if c.startswith("cv_")]].round(3).to_string(index=False))

    if args.compare:
        comp = pd.DataFrame([r for t in args.tasks for r in compare_models(t, args.split)])
        comp.to_csv(out / "model_comparison.csv", index=False)
        print(comp.pivot(index="model", columns="task", values="R2").round(3).to_string())


if __name__ == "__main__":
    main()
