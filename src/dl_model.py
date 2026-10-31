"""Exploratory deep learning model for organic carbon (Figures 5-6).

Feed-forward neural network predicting organic carbon concentration (g/kg)
for the 0-30 cm layer, following the original development notebook
(tierrafoods-hud/notebooks: analysis/deeplearning-evaluation.ipynb).

The input dataset has exactly one row per soil profile (single harmonised
0-30 cm layer), so the random 80/20 split is also a profile-level split.
Inputs are clay, sand, silt and pH. Seeds are fixed for repeatability.

Usage
-----
    python src/dl_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "Mexico_single_depth_orgc_merged.csv"
OUT_DIR = ROOT / "outputs" / "dl"

SEED = 42
FEATURES = ["clay", "phaq", "sand", "silt"]
TARGET = "orgc"


def build_model(n_inputs: int) -> tf.keras.Model:
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(n_inputs,)),
        layers.Dense(64, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(32, activation="relu"), layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(16, activation="relu"), layers.BatchNormalization(),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def plot_history(history: dict, path: Path) -> None:
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, key, title, ylabel in [(axes[0], "loss", "Loss (MSE)", "Loss"),
                                       (axes[1], "mae", "Mean Absolute Error", "MAE")]:
            ax.plot(history[key], label=f"Training {ylabel if key == 'mae' else 'Loss'}")
            ax.plot(history[f"val_{key}"], label=f"Validation {ylabel if key == 'mae' else 'Loss'}")
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def plot_predictions(y_true, y_pred, stats: dict, path: Path) -> None:
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        ax.scatter(y_true, y_pred, alpha=0.5, s=25, color="steelblue", edgecolors="none",
                   label="Predictions")
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect fit")
        ax.text(0.98, 0.04,
                f"MSE: {stats['MSE']:9.4f}\nRMSE: {stats['RMSE']:8.4f}\nMAE: {stats['MAE']:9.4f}\n"
                f"R²: {stats['R2']:10.4f}",
                transform=ax.transAxes, ha="right", va="bottom", family="monospace",
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "grey"})
        ax.set_xlabel("Actual Organic Carbon (g/kg)")
        ax.set_ylabel("Predicted Organic Carbon (g/kg)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass

    df = pd.read_csv(DATA_FILE)
    assert df["profile_id"].is_unique, "expected one row per profile"
    df = df[FEATURES + [TARGET]].dropna()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    scaler = StandardScaler().fit(X_train)
    model = build_model(len(FEATURES))
    history = model.fit(
        scaler.transform(X_train), y_train, validation_split=0.2, epochs=200, batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                                    restore_best_weights=True)],
        verbose=0,
    )
    pred = model.predict(scaler.transform(X_test), verbose=0).ravel()

    mse = mean_squared_error(y_test, pred)
    stats = {
        "n_profiles": len(df), "n_train": len(X_train), "n_test": len(X_test),
        "epochs_trained": len(history.history["loss"]),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "best_val_MAE": float(history.history["val_mae"][int(np.argmin(history.history["val_loss"]))]),
        "MSE": mse, "RMSE": float(np.sqrt(mse)), "MAE": mean_absolute_error(y_test, pred),
        "R2": r2_score(y_test, pred),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dl_metrics.json").write_text(json.dumps(stats, indent=2))
    plot_history(history.history, OUT_DIR / "organic_carbon_dl_training_history.pdf")
    plot_predictions(y_test.to_numpy(), pred, stats, OUT_DIR / "organic_carbon_dl_actual_vs_predicted.pdf")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
