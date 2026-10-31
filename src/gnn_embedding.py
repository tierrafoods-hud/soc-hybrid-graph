"""Graph neural network soil embeddings and soil archetypes (Figure 7, Tables 2-3).

Script version of the development notebook (tierrafoods-hud/causal_graph:
gnn_embedding.ipynb) with fixed random seeds.

* Nodes: one per harmonised profile layer (notebooks/dataset.py).
* Node features: standardised numeric soil and terrain variables, one-hot encoded
  land cover, soil type, ecoregion, bedrock and depth layer, plus latitude and
  longitude. Organic (``orgc``) and inorganic (``inorgc``) carbon are excluded
  unless ``--include-carbon`` is given.
* Edges: undirected, between nodes whose Haversine distance is <= 10 km.
* Model: two GCNConv layers (128 -> 64) with ReLU, trained for 50 epochs with
  Adam (lr 0.01) and a triplet margin loss (margin 1.0). Anchor/positive pairs
  are graph neighbours, negatives are random non-neighbours.
* Clustering: K-Means (k = 8) on the 64-dimensional embeddings.

Usage
-----
    python src/gnn_embedding.py                    # carbon excluded from node features
    python src/gnn_embedding.py --include-carbon   # carbon included (original notebook set-up)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import df as DATA  # noqa: E402

SEED = 42
NEIGHBOR_RADIUS_KM = 10.0
HIDDEN_DIM, EMBEDDING_DIM = 128, 64
LEARNING_RATE, EPOCHS, TRIPLET_MARGIN = 0.01, 50, 1.0
N_CLUSTERS = 8

NUMERIC = ["sand", "elcosp", "slope", "silt_clay", "phaq", "rock_fragments"]
CARBON = ["orgc", "inorgc"]
CATEGORICAL = ["landcover", "soil_type", "ecoregion", "bedrock", "depth"]
MEXICO_GEOJSON = ROOT / "data" / "mexico.geojson"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def node_features(df: pd.DataFrame, include_carbon: bool) -> torch.Tensor:
    numeric = NUMERIC + (CARBON if include_carbon else [])
    pre = ColumnTransformer(
        [("num", StandardScaler(), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
         ("coords", "passthrough", ["latitude", "longitude"])])
    x = pre.fit_transform(df)
    x = x.toarray() if hasattr(x, "toarray") else x
    return torch.tensor(x, dtype=torch.float)


def proximity_edges(df: pd.DataFrame, radius_km: float) -> torch.Tensor:
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    dist = haversine_distances(coords) * 6371.0
    i, j = np.where(np.triu(dist <= radius_km, k=1))
    edges = np.concatenate([np.stack([i, j]), np.stack([j, i])], axis=1)
    return torch.tensor(edges, dtype=torch.long)


class GCNEmbedder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, HIDDEN_DIM)
        self.conv2 = GCNConv(HIDDEN_DIM, EMBEDDING_DIM)

    def forward(self, x, edge_index):
        return self.conv2(torch.relu(self.conv1(x, edge_index)), edge_index)


def train(model: GCNEmbedder, x: torch.Tensor, edge_index: torch.Tensor) -> list[float]:
    n = x.shape[0]
    adj = [set() for _ in range(n)]
    for u, v in edge_index.t().tolist():
        adj[u].add(v)
    adj_list = [sorted(a) for a in adj]
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN)
    losses = []
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        anchors, positives, negatives = [], [], []
        for a in range(n):
            if not adj_list[a]:
                continue
            p = random.choice(adj_list[a])
            while True:
                q = random.randint(0, n - 1)
                if q != a and q not in adj[a]:
                    break
            anchors.append(a); positives.append(p); negatives.append(q)
        emb = model(x, edge_index)
        loss = loss_fn(emb[anchors], emb[positives], emb[negatives])
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def plot_map(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    if MEXICO_GEOJSON.exists():
        gj = json.loads(MEXICO_GEOJSON.read_text(encoding="utf-8"))
        for feat in gj["features"]:
            geom = feat["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                ring = np.asarray(poly[0])
                ax.plot(ring[:, 0], ring[:, 1], color="black", lw=0.5)
    cmap = plt.get_cmap("tab10")
    for k in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == k]
        ax.scatter(sub["longitude"], sub["latitude"], s=10, color=cmap(k), label=f"Cluster {k}")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8, markerscale=1.5, frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-carbon", action="store_true")
    args = parser.parse_args()
    set_seeds(SEED)

    df = DATA.reset_index(drop=True).copy()
    x = node_features(df, args.include_carbon)
    edge_index = proximity_edges(df, NEIGHBOR_RADIUS_KM)
    model = GCNEmbedder(x.shape[1])
    losses = train(model, x, edge_index)
    model.eval()
    with torch.no_grad():
        emb = model(x, edge_index).numpy()

    df["cluster"] = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10).fit_predict(emb)
    mode = lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan  # noqa: E731
    numeric_stats = df.groupby("cluster")[NUMERIC[:3] + CARBON + NUMERIC[3:]].agg(["mean", "std"])
    categorical_stats = df.groupby("cluster")[["landcover", "bedrock", "soil_type", "ecoregion"]].agg(mode)
    counts = df["cluster"].value_counts().sort_index()

    out = ROOT / "outputs" / "gnn" / ("with_carbon" if args.include_carbon else "carbon_excluded")
    out.mkdir(parents=True, exist_ok=True)
    numeric_stats.round(3).to_csv(out / "cluster_numeric_stats.csv")
    categorical_stats.to_csv(out / "cluster_categorical_modes.csv")
    df[["latitude", "longitude", "depth", "cluster"]].to_csv(out / "cluster_assignments.csv", index=False)
    summary = {
        "nodes": int(x.shape[0]), "node_features": int(x.shape[1]),
        "undirected_edges": int(edge_index.shape[1] // 2),
        "isolated_nodes": int(x.shape[0] - len(set(edge_index[0].tolist()))),
        "final_triplet_loss": losses[-1], "silhouette_k8": float(silhouette_score(emb, df["cluster"])),
        "cluster_sizes": {int(k): int(v) for k, v in counts.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_map(df, out / "soil_clusters_map.pdf")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(json.dumps(summary, indent=2))
        print(numeric_stats.round(2))
        print(categorical_stats)


if __name__ == "__main__":
    main()
