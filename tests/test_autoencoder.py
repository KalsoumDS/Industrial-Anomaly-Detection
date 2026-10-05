"""
Unit tests for Industrial-Anomaly-Detection.

All tests use synthetic data — no network access required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def small_df() -> pd.DataFrame:
    """200-sample dataset with 16 injected anomalies."""
    rng = np.random.default_rng(42)
    n = 200
    timestamps = pd.date_range("2024-01-01", periods=n, freq="15s")
    is_anomaly = np.zeros(n, dtype=int)
    is_anomaly[rng.choice(n, size=16, replace=False)] = 1
    vibration = rng.normal(2.0, 0.4, n)
    vibration[is_anomaly == 1] += 4.0
    return pd.DataFrame({
        "timestamp":           timestamps,
        "vibration_mm_s":      vibration,
        "temperature_celsius": rng.normal(75.0, 3.0, n),
        "pressure_bar":        rng.normal(4.2,  0.3, n),
        "flow_rate_l_min":     rng.normal(120.0, 8.0, n),
        "is_anomaly":          is_anomaly,
    })


class TestIoTAutoencoder:

    def test_forward_pass_preserves_shape(self):
        """Output tensor must have the same shape as the input."""
        from offline_validate import IoTAutoencoder
        model = IoTAutoencoder(input_dim=4)
        x = torch.randn(32, 4)
        assert model(x).shape == x.shape

    def test_reconstruction_mse_non_negative(self):
        """Per-sample MSE must be >= 0 by definition."""
        from offline_validate import IoTAutoencoder
        model = IoTAutoencoder(input_dim=4)
        model.eval()
        with torch.no_grad():
            x = torch.randn(64, 4)
            mse = ((x - model(x)) ** 2).mean(dim=1)
        assert (mse >= 0).all()

    def test_training_does_not_produce_nan(self, small_df):
        """10 training epochs on clean data must not produce NaN loss."""
        from offline_validate import IoTAutoencoder, FEATURE_COLS
        X = small_df[FEATURE_COLS].values.astype(np.float32)
        mean, std = X.mean(0), X.std(0) + 1e-8
        X_t = torch.tensor((X - mean) / std)
        model = IoTAutoencoder(input_dim=4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()
        for _ in range(10):
            opt.zero_grad()
            loss = crit(model(X_t), X_t)
            loss.backward()
            opt.step()
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_anomaly_mse_exceeds_normal_mse(self, small_df, tmp_path):
        """Injected anomalies must produce higher average reconstruction error."""
        from offline_validate import train_autoencoder
        model_path = str(tmp_path / "ae.pt")
        _, scores = train_autoencoder(small_df, epochs=30, model_path=model_path)
        normal_mse  = scores[small_df["is_anomaly"] == 0].mean()
        anomaly_mse = scores[small_df["is_anomaly"] == 1].mean()
        assert anomaly_mse > normal_mse, (
            f"Anomaly MSE ({anomaly_mse:.4f}) <= normal MSE ({normal_mse:.4f})"
        )

    def test_model_weights_persisted(self, small_df, tmp_path):
        """torch.save must produce a loadable checkpoint with 'state_dict' key."""
        from offline_validate import train_autoencoder, IoTAutoencoder
        model_path = str(tmp_path / "ae.pt")
        train_autoencoder(small_df, epochs=5, model_path=model_path)
        checkpoint = torch.load(model_path, weights_only=False)
        assert "state_dict" in checkpoint
        model = IoTAutoencoder(input_dim=4)
        model.load_state_dict(checkpoint["state_dict"])


class TestEvaluationMetrics:

    def test_roc_auc_above_random(self, small_df, tmp_path):
        """Trained model must achieve ROC-AUC > 0.5."""
        from offline_validate import train_autoencoder, evaluate
        _, scores = train_autoencoder(small_df, epochs=30, model_path=str(tmp_path / "ae.pt"))
        m = evaluate(small_df["is_anomaly"].values, scores)
        assert m["roc_auc"] > 0.5, f"ROC-AUC {m['roc_auc']:.4f} <= 0.5"

    def test_pr_auc_strictly_positive(self, small_df, tmp_path):
        """PR-AUC must be > 0."""
        from offline_validate import train_autoencoder, evaluate
        _, scores = train_autoencoder(small_df, epochs=30, model_path=str(tmp_path / "ae.pt"))
        m = evaluate(small_df["is_anomaly"].values, scores)
        assert m["pr_auc"] > 0.0

    def test_best_f1_in_unit_interval(self, small_df, tmp_path):
        """Best F1 must lie in [0, 1]."""
        from offline_validate import train_autoencoder, evaluate
        _, scores = train_autoencoder(small_df, epochs=30, model_path=str(tmp_path / "ae.pt"))
        m = evaluate(small_df["is_anomaly"].values, scores)
        assert 0.0 <= m["best_f1"] <= 1.0

    def test_metrics_dict_has_all_keys(self, small_df, tmp_path):
        """evaluate() must return all expected metric keys."""
        from offline_validate import train_autoencoder, evaluate
        _, scores = train_autoencoder(small_df, epochs=5, model_path=str(tmp_path / "ae.pt"))
        m = evaluate(small_df["is_anomaly"].values, scores)
        required = {"roc_auc", "pr_auc", "best_f1", "best_threshold",
                    "precision_at_best_f1", "recall_at_best_f1"}
        assert required <= m.keys()
