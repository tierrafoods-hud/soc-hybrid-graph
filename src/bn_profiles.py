"""Bayesian network environmental profiles for high and low organic carbon (Table 4).

Script version of notebooks/02_bayesian_network_profiles.ipynb. It uses the same
network structure, discretisation and estimator, and queries the network for both
``orgc = high`` and ``orgc = low`` (the notebook shows only the high case).

Discretisation: continuous variables are split into low/medium/high terciles
(``pandas.qcut``). Sand uses fixed classes (<20, 20-45, >=45 %). Soil type, bedrock,
depth layer, land cover and ecoregion are treated as categorical.

Usage
-----
    python src/bn_profiles.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import df  # noqa: E402  (shared input table, one row per profile layer)

OUT_DIR = ROOT / "outputs" / "bn"

EDGES = [
    ("sand", "elcosp"),
    ("slope", "landcover"), ("slope", "orgc"), ("slope", "inorgc"),
    ("silt_clay", "elcosp"), ("silt_clay", "phaq"),
    ("bedrock", "soil_type"),
    ("soil_type", "silt_clay"), ("soil_type", "phaq"), ("soil_type", "elcosp"),
    ("elcosp", "inorgc"),
    ("landcover", "elcosp"), ("landcover", "orgc"),
    ("phaq", "elcosp"), ("phaq", "orgc"), ("phaq", "inorgc"),
    ("ecoregion", "landcover"), ("ecoregion", "orgc"), ("ecoregion", "phaq"), ("ecoregion", "inorgc"),
    ("rock_fragments", "orgc"), ("rock_fragments", "inorgc"),
    ("depth", "rock_fragments"), ("depth", "orgc"), ("depth", "inorgc"),
]
CATEGORICAL = ["soil_type", "bedrock", "depth", "landcover", "ecoregion"]
SAND_BINS = [0, 20, 45, float("inf")]
LABELS = ["low", "medium", "high"]

# Names used when reporting the categorical states.
IGBP = {1: "Evergreen needleleaf forests", 2: "Evergreen broadleaf forests",
        3: "Deciduous needleleaf forests", 4: "Deciduous broadleaf forests", 5: "Mixed forests",
        6: "Closed shrublands", 7: "Open shrublands", 8: "Woody savannas", 9: "Savannas",
        10: "Grasslands", 11: "Permanent wetlands", 12: "Croplands", 13: "Urban and built-up",
        14: "Cropland/natural vegetation mosaics", 15: "Snow and ice", 16: "Barren",
        17: "Water bodies"}


def ecoregion_names() -> dict:
    """Map ecoregion zone numbers to CONABIO zone names from the input data."""
    src = pd.read_csv(ROOT / "data" / "processed" / "Mexico_standardized_cfvo_orgc.csv",
                      usecols=["zone_number", "zone_name"]).dropna().drop_duplicates()
    return dict(zip(src["zone_number"], src["zone_name"]))


def discretise(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    nodes = {v for e in EDGES for v in e}
    out, bins_info = data.copy(), {}
    for var in sorted(nodes - set(CATEGORICAL)):
        if var == "sand":
            out[var] = pd.cut(data[var], bins=SAND_BINS, labels=LABELS, include_lowest=True, right=False)
            bins_info[var] = SAND_BINS
        else:
            try:
                out[var], bins = pd.qcut(data[var], q=3, labels=LABELS, duplicates="drop", retbins=True)
            except ValueError:
                # Too many tied values for terciles (e.g. inorgc, mostly zero):
                # fall back to three equal-width classes, as in the notebook.
                out[var], bins = pd.cut(data[var], bins=3, labels=LABELS, retbins=True)
            bins_info[var] = list(bins)
    return out, bins_info


def main() -> None:
    data, bins_info = discretise(df)
    model = BayesianNetwork(EDGES)
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    inference = VariableElimination(model)

    eco, rows = ecoregion_names(), []
    for level in ("high", "low"):
        for var in model.nodes():
            if var == "orgc":
                continue
            res = inference.query(variables=[var], evidence={"orgc": level}, show_progress=False)
            for state, p in zip(res.state_names[var], res.values.ravel()):
                label = state
                if var == "landcover":
                    label = IGBP.get(int(float(state)), state)
                elif var == "ecoregion":
                    label = eco.get(float(state), state)
                rows.append({"orgc": level, "variable": var, "state": state, "label": label,
                             "probability": float(p)})
    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "bn_orgc_profiles.csv", index=False)
    pd.DataFrame([{"variable": k, "bins": v} for k, v in bins_info.items()]) \
        .to_csv(OUT_DIR / "bn_discretisation.csv", index=False)

    summary_vars = ["phaq", "depth", "ecoregion", "landcover", "slope", "rock_fragments", "bedrock", "soil_type"]
    print(f"pH tercile limits: {bins_info['phaq'][1]:.2f} / {bins_info['phaq'][2]:.2f}")
    for level in ("high", "low"):
        print(f"\n=== orgc = {level} ===")
        sub = table[table["orgc"] == level]
        for var in summary_vars:
            top = sub[sub["variable"] == var].nlargest(3, "probability")
            print(f"{var:15s} " + "; ".join(f"{r.label} {r.probability:.1%}" for r in top.itertuples()))


if __name__ == "__main__":
    main()
