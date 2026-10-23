"""Shared input table for the graph-based notebooks (GNN embedding and Bayesian network).

Merges the harmonised organic carbon (orgc) and calcium carbonate equivalent
(tceq, used here as inorganic carbon) datasets on profile, depth layer and date.
Each row is one depth layer (0-5, 5-15 or 15-30 cm) of one soil profile.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

orgc_df = pd.read_csv(DATA_DIR / "Mexico_standardized_cfvo_orgc.csv")

tceq_df = pd.read_csv(DATA_DIR / "Mexico_standardized_cfvo_tceq.csv")

df = pd.merge(orgc_df, tceq_df[['profile_id','depth_category','date','tceq']], how='left', on=['profile_id', 'depth_category', 'date']).dropna()

df['silt_clay'] = df['silt'] + df['clay']

# rename slopemean to slope
df = df.rename(columns={"slopemean": "slope", "tceq": "inorgc", "cfr":"rock_fragments", "depth_category":"depth", "zone_number":"ecoregion"})

# drop na from specific columns
df = df[["latitude","longitude","sand",
    "elcosp",
    "slope",
    "landcover",
    "orgc",
    "inorgc",
    "silt_clay",
    "phaq",
    "bedrock",
    "soil_type",
    "ecoregion",
    "rock_fragments",
    "depth"]]

df = df.dropna()