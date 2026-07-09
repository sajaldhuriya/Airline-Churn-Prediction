"""
Trains an SVC inside a sklearn Pipeline(StandardScaler -> SVC), then logs
the WHOLE pipeline to MLflow along with the exact feature-name list.

This is the production training entry point — the DAG and `make retrain`
both invoke this script.

Why a Pipeline?
  The notebook (`notebooks/machine_learning.ipynb`) showed the SVC
  winner is {kernel=rbf, C=1.0, class_weight=balanced} on the engagement-
  only feature set (4 numerics + one-hot dummies), with holdout recall
  0.847. If we save just the SVC, the API would have to re-implement
  scaling and would silently break when the scaler drifts. A Pipeline
  guarantees the same StandardScaler that was fit on X_train is the one
  used at inference.

Why rbf, not linear?
  The notebook's arena section (`notebooks/machine_learning.ipynb` §5-6)
  compared linear and rbf kernels on the engagement-only feature set.
  The rbf kernel won on holdout recall (0.847 vs 0.838 for linear) and
  on CV recall (0.845 vs 0.835). The hyperparameter grid was the same
  `clf__C: loguniform(0.1, 100)` from `configs/model_params.yaml`.

What gets logged to MLflow:
  - params: kernel, C, class_weight, feature_count, scaler_mean, scaler_scale
  - metrics: accuracy, precision, recall, f1, roc_auc, plus the full
    confusion matrix (tn, fp, fn, tp) for monitoring drift over time
  - artifact 'tuned_model': the fitted Pipeline
  - artifact 'feature_names.txt': the exact ordered 19-column schema
  - artifact 'permutation_importances.json': model's view of which
    features matter, so the dashboard can render "drivers of churn"
    without depending on SVC.coef_ (rbf kernels don't expose it)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data_pipeline.pull_snowflake import fetch_and_clean_data  # noqa: E402

logger = logging.getLogger(__name__)


# Champion hyperparameters — pinned from
# `notebooks/machine_learning.ipynb` §5-6 (randomized search, n_iter=8,
# cv=3, scoring=recall). The arena settled on rbf/C≈1.0 as the winner.
CHAMPION_KERNEL = "rbf"
CHAMPION_C = 1.0
CHAMPION_CLASS_WEIGHT = "balanced"
RANDOM_STATE = 42


def _resolve_mlflow_uri() -> str:
    """Always use an absolute SQLite path, regardless of CWD."""
    db_path = os.path.abspath(os.path.join(ROOT, "mlflow.db"))
    return f"sqlite:///{db_path}"


def _build_pipeline() -> Pipeline:
    """Build (but do not fit) the production Pipeline. Exposed for tests."""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=CHAMPION_KERNEL,
            C=CHAMPION_C,
            class_weight=CHAMPION_CLASS_WEIGHT,
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])


def _permutation_importance_payload(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 20,
    seed: int = RANDOM_STATE,
) -> List[Dict[str, float]]:
    """Compute permutation importances for the fitted pipeline.

    The notebook (section 8) uses this to compare the model's view of
    feature importance against the stats battery. The dashboard reads
    the same artifact so it can render "drivers of churn" without
    relying on `pipeline.coef_` (rbf kernels don't expose coefficients).

    Returns a list of {feature, importance_mean, importance_std} dicts,
    sorted by importance_mean descending.
    """
    result = permutation_importance(
        pipeline, X, y,
        n_repeats=n_repeats, random_state=seed, scoring="recall",
    )
    df = pd.DataFrame({
        "feature":          list(X.columns),
        "importance_mean":  result.importances_mean.astype(float),
        "importance_std":   result.importances_std.astype(float),
    }).sort_values("importance_mean", ascending=False)
    return df.to_dict(orient="records")


def train_and_log_model() -> Dict[str, float]:
    logger.info("Starting modular SVC training...")

    # 1) Pull data (returns the feature-name list as the third element)
    X, y, feature_names = fetch_and_clean_data()
    logger.info("Feature schema (%d cols): %s", len(feature_names), feature_names)

    # 2) Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    # 3) The Pipeline: scaler + SVC. The scaler is fit only on X_train.
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=CHAMPION_KERNEL,
            C=CHAMPION_C,
            class_weight=CHAMPION_CLASS_WEIGHT,
            probability=True,   # enables predict_proba
            random_state=RANDOM_STATE,
        )),
    ])

    # 4) MLflow setup with an absolute tracking URI
    mlflow.set_tracking_uri(_resolve_mlflow_uri())
    mlflow.set_experiment("Airline_Churn_Production")

    with mlflow.start_run(run_name="SVC_Champion") as run:
        params = {
            "kernel":       CHAMPION_KERNEL,
            "C":            float(CHAMPION_C),
            "class_weight": CHAMPION_CLASS_WEIGHT,
            "feature_count": len(feature_names),
        }
        mlflow.log_params(params)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        # SVC with probability=True is calibrated via Platt scaling,
        # so predict_proba is well-defined.
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics.update({
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_positives": float(tp),
        })
        mlflow.log_metrics(metrics)

        # 5) Log the entire Pipeline, not just the SVC
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="tuned_model",
            input_example=X_test.iloc[:2],
        )

        # 6) Save the feature-name list as a sidecar text artifact so the
        #    API can read it back at startup, eliminating hard-coded order
        #    drift between training and serving.
        mlflow.log_text("\n".join(feature_names), "feature_names.txt")

        # 7) Permutation importances. The rbf kernel has no coef_ we
        #    could surface directly; this is the model's *own* view of
        #    which features matter. The dashboard reads this artifact
        #    to render the "drivers of churn" donut.
        importances = _permutation_importance_payload(pipeline, X_test, y_test)
        mlflow.log_text(
            json.dumps(importances, indent=2),
            "permutation_importances.json",
        )

        logger.info(
            "Champion trained. recall=%.3f, precision=%.3f, f1=%.3f, auc=%.3f",
            metrics["recall"], metrics["precision"],
            metrics["f1_score"], metrics["roc_auc"],
        )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    train_and_log_model()
