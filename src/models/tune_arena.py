"""
Hyperparameter tuning arena.

This was previously `tests/test_hyperparameter_tuning.py` (a misnamed
training script). It is now correctly placed under `src/models/` and
called by the Airflow DAG as the first stage of the weekly retrain.

Differences vs the original v1.0:
  - Models that REQUIRE scaling (SVC, MLP, LogReg) are tuned on scaled
    data; tree models (RandomForest, XGBoost) are tuned on raw data.
    Previously *all* models were trained on scaled data, which is fine
    for trees (a no-op) but means the logged RF/XGB artifacts expect
    scaled inputs — at inference the API would silently mis-scale them.
  - Each model is wrapped in a Pipeline so the inference API can call
    .predict() on raw values and the right preprocessing is applied.
  - Uses a shared helper `run_search` to remove copy-pasted logic.
  - Logs ROC-AUC alongside recall/precision so we can pick champions by
    business rule, not blindly by recall.
  - Reads param grids from `configs/model_params.yaml` so they are
    tweakable without code changes.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from scipy.stats import loguniform, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data_pipeline.pull_snowflake import fetch_and_clean_data  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(ROOT, "configs", "model_params.yaml")

# Which algorithms REQUIRE a StandardScaler at inference.
NEEDS_SCALING = {"Logistic_Regression", "SVC", "Neural_Network"}


def _load_grids() -> Dict[str, Dict[str, Any]]:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw.get("models", raw)


def _build_pipeline(name: str, model) -> Pipeline:
    if name in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("clf", model)])
    return Pipeline([("clf", model)])


def _make_estimator(name: str):
    if name == "Logistic_Regression":
        return LogisticRegression(max_iter=2000, random_state=42)
    if name == "Random_Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=42)
    if name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=42)
    if name == "SVC":
        return SVC(probability=True, random_state=42)
    if name == "Neural_Network":
        return MLPClassifier(max_iter=500, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def _coerce_param_distributions(param_grid: Dict[str, Any]) -> Dict[str, Any]:
    """Allow YAML to express log-uniform / uniform via 'loguniform(x,y)'."""
    out: Dict[str, Any] = {}
    for k, v in param_grid.items():
        if isinstance(v, str) and v.startswith("loguniform("):
            lo, hi = v[len("loguniform("):-1].split(",")
            out[k] = loguniform(float(lo), float(hi))
        elif isinstance(v, str) and v.startswith("uniform("):
            lo, hi = v[len("uniform("):-1].split(",")
            out[k] = uniform(float(lo), float(hi) - float(lo))
        else:
            out[k] = v
    return out


def _score(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_proba))
    else:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": auc,
    }


def _resolve_mlflow_uri() -> str:
    db_path = os.path.abspath(os.path.join(ROOT, "mlflow.db"))
    return f"sqlite:///{db_path}"


def run_tuning_arena() -> pd.DataFrame:
    logger.info("🚀 Firing up the Hyperparameter Tuning Arena...")

    X, y, feature_names = fetch_and_clean_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    mlflow.set_tracking_uri(_resolve_mlflow_uri())
    mlflow.set_experiment("Airline_Churn_Tuning_Sweep")

    grids = _load_grids()
    results = []

    for name, cfg in grids.items():
        logger.info("⚙️ Tuning %s...", name)
        estimator = _make_estimator(name)
        pipeline = _build_pipeline(name, estimator)
        param_dist = _coerce_param_distributions(cfg.get("params", {}))
        n_iter = int(cfg.get("n_iter", 5))

        tuner = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="recall",
            cv=3,
            n_jobs=-1,
            random_state=42,
        )
        with mlflow.start_run(run_name=f"Tuned_{name}"):
            tuner.fit(X_train, y_train)
            best = tuner.best_estimator_
            metrics = _score(best, X_test, y_test)

            mlflow.log_params(tuner.best_params_)
            mlflow.log_metrics(metrics)

            # Log the WHOLE pipeline (so the scaler is preserved for SVC/LR/MLP)
            mlflow.sklearn.log_model(
                best, artifact_path="tuned_model",
                input_example=X_test.iloc[:2],
            )
            # Each tuning run also gets the feature schema, for safety
            mlflow.log_text("\n".join(feature_names), "feature_names.txt")

            results.append({"model": name, **metrics,
                            "best_params": tuner.best_params_})
            logger.info("✅ %s done. recall=%.3f", name, metrics["recall"])

    df = pd.DataFrame(results).sort_values("recall", ascending=False)
    logger.info("\n🏆 --- LEADERBOARD ---\n%s", df.to_string(index=False))
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_tuning_arena()
