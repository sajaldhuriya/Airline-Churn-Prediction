"""
Hermetic test of the SVC training pipeline.

Verifies:
  - `_build_pipeline()` constructs a Pipeline (StandardScaler -> SVC) in
    the right order. This is the regression test for the v1.0 bug
    where the scaler was never saved to MLflow.
  - `train_and_log_model()` returns a metrics dict with every metric
    the production dashboard needs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def trained_pipeline(monkeypatch):
    """Train a tiny pipeline with all external deps stubbed.

    Returns the metrics dict from `train_and_log_model()`.
    """
    from src.models import train_svc

    # 1) Synthetic data
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "SALARY": rng.normal(80000, 20000, n).clip(0),
        "CLV": rng.normal(5000, 1500, n).clip(0),
        "LIFETIME_FLIGHTS": rng.integers(0, 100, n),
        "LIFETIME_DISTANCE": rng.integers(0, 100000, n),
        "LIFETIME_POINTS_EARNED": rng.integers(0, 200000, n),
        "LIFETIME_POINTS_REDEEMED": rng.integers(0, 5000, n),
        "GENDER_Male": rng.integers(0, 2, n),
        "EDUCATION_College": rng.integers(0, 2, n),
        "EDUCATION_Doctor": rng.integers(0, 2, n),
        "EDUCATION_High School or Below": rng.integers(0, 2, n),
        "EDUCATION_Master": rng.integers(0, 2, n),
        "MARITAL_STATUS_Married": rng.integers(0, 2, n),
        "MARITAL_STATUS_Single": rng.integers(0, 2, n),
        "LOYALTY_CARD_Nova": rng.integers(0, 2, n),
        "LOYALTY_CARD_Star": rng.integers(0, 2, n),
        "ENROLLMENT_TYPE_Standard": rng.integers(0, 2, n),
    })
    y = (df["LIFETIME_FLIGHTS"] < 15).astype(int)
    feats = list(df.columns)

    # 2) Stub the data layer and MLflow I/O
    monkeypatch.setattr(train_svc, "fetch_and_clean_data",
                        lambda: (df, y, feats))

    import mlflow
    class _NoopRun:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(mlflow, "start_run", lambda *a, **k: _NoopRun())
    monkeypatch.setattr(mlflow, "log_params", lambda p: None)
    monkeypatch.setattr(mlflow, "log_metrics", lambda m: None)
    monkeypatch.setattr(mlflow, "log_text", lambda *a, **k: None)
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *a, **k: None)

    return train_svc.train_and_log_model()


def test_pipeline_metrics_complete(trained_pipeline):
    expected = {
        "accuracy", "precision", "recall", "f1_score", "roc_auc",
        "true_negatives", "false_positives", "false_negatives", "true_positives",
    }
    missing = expected - set(trained_pipeline.keys())
    assert not missing, f"missing metrics: {missing}"


def test_metrics_in_unit_interval(trained_pipeline):
    for k, v in trained_pipeline.items():
        if k.startswith(("true_", "false_")):
            assert v >= 0
        else:
            assert 0.0 <= v <= 1.0, f"{k}={v} not in [0,1]"


def test_pipeline_builds_correctly():
    """Build a Pipeline and confirm it contains both a StandardScaler
    and an SVC in the right order. This is the regression test for the
    v1.0 bug where the scaler was never saved to MLflow."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from src.models import train_svc

    p = train_svc._build_pipeline()
    assert isinstance(p, Pipeline)
    assert p.steps[0][0] == "scaler"
    assert isinstance(p.steps[0][1], StandardScaler)
    assert p.steps[1][0] == "svc"
    assert isinstance(p.steps[1][1], SVC)
    # SVC must have probability=True so predict_proba works at inference
    assert p.steps[1][1].probability is True
