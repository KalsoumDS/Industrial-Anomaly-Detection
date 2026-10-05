"""
offline_validate.py — Offline model evaluation for IoTAutoencoder.

Computes ROC-AUC, PR-AUC, and optimal F1 threshold on a SKAB-style
benchmark with controlled anomaly injection. Runs independently of
the Streamlit UI.

Usage:
    python offline_validate.py [--epochs 60] [--n-samples 2000]
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class IoTAutoencoder(nn.Module):
    """Compact deep autoencoder for multivariate IoT telemetry (4 channels)."""

    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_benchmark_data(n_samples: int = 2000, anomaly_ratio: float = 0.08) -> pd.DataFrame:
    """
    SKAB-style benchmark: inject bearing friction drift anomalies.

    Anomaly regions exhibit elevated vibration (+4 sigma) and temperature
    (+15 C) to simulate imminent mechanical failure.
    """
    rng = np.random.default_rng(seed=42)
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="15s")

    vibration    = rng.normal(2.0,   0.4,  n_samples)
    temperature  = rng.normal(75.0,  3.0,  n_samples)
    pressure     = rng.normal(4.2,   0.3,  n_samples)
    flow_rate    = rng.normal(120.0, 8.0,  n_samples)

    is_anomaly = np.zeros(n_samples, dtype=int)
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_idx = rng.choice(n_samples, size=n_anomalies, replace=False)
    is_anomaly[anomaly_idx] = 1

    vibration[anomaly_idx]   += rng.normal(4.0,  0.5, n_anomalies)
    temperature[anomaly_idx] += rng.normal(15.0, 2.0, n_anomalies)

    return pd.DataFrame({
        "timestamp":           timestamps,
        "vibration_mm_s":      vibration,
        "temperature_celsius": temperature,
        "pressure_bar":        pressure,
        "flow_rate_l_min":     flow_rate,
        "is_anomaly":          is_anomaly,
    })


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

FEATURE_COLS = ["vibration_mm_s", "temperature_celsius", "pressure_bar", "flow_rate_l_min"]


def train_autoencoder(
    df: pd.DataFrame,
    epochs: int = 60,
    model_path: str = "models/autoencoder.pt",
) -> tuple[IoTAutoencoder, np.ndarray]:
    """Train on normal samples only (unsupervised). Persist weights to disk."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    normal_mask = df["is_anomaly"].values == 0
    X_train = torch.tensor(X_norm[normal_mask])

    model = IoTAutoencoder(input_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_train), X_train)
        loss.backward()
        optimizer.step()

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std}, model_path)
    print(f"  Model saved -> {model_path}")

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_norm)
        recon = model(X_tensor)
        mse_scores = ((X_tensor - recon) ** 2).mean(dim=1).numpy()

    return model, mse_scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Compute ROC-AUC, PR-AUC, and the threshold that maximises F1."""
    roc_auc = roc_auc_score(labels, scores)
    pr_auc  = average_precision_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx  = int(np.argmax(f1_scores))

    return {
        "roc_auc":              float(roc_auc),
        "pr_auc":               float(pr_auc),
        "best_f1":              float(f1_scores[best_idx]),
        "best_threshold":       float(thresholds[min(best_idx, len(thresholds) - 1)]),
        "precision_at_best_f1": float(precision[best_idx]),
        "recall_at_best_f1":    float(recall[best_idx]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline validation — IoTAutoencoder")
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--n-samples",     type=int,   default=2000)
    parser.add_argument("--anomaly-ratio", type=float, default=0.08)
    parser.add_argument("--model-path",    type=str,   default="models/autoencoder.pt")
    args = parser.parse_args()

    print("Generating SKAB-style benchmark...")
    df = generate_benchmark_data(n_samples=args.n_samples, anomaly_ratio=args.anomaly_ratio)
    n_anom = int(df["is_anomaly"].sum())
    print(f"  {len(df)} samples | {n_anom} anomalies ({n_anom / len(df) * 100:.1f}%)")

    print(f"Training autoencoder ({args.epochs} epochs)...")
    _, mse_scores = train_autoencoder(df, epochs=args.epochs, model_path=args.model_path)

    print("Evaluating...")
    m = evaluate(df["is_anomaly"].values, mse_scores)

    print()
    print("=" * 50)
    print("  OFFLINE VALIDATION RESULTS")
    print("=" * 50)
    print(f"  ROC-AUC              : {m['roc_auc']:.4f}")
    print(f"  PR-AUC               : {m['pr_auc']:.4f}")
    print(f"  Best F1              : {m['best_f1']:.4f}")
    print(f"  Optimal threshold    : {m['best_threshold']:.6f}")
    print(f"  Precision @ best F1  : {m['precision_at_best_f1']:.4f}")
    print(f"  Recall    @ best F1  : {m['recall_at_best_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
