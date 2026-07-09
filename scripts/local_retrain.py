"""
Local retrain: runs the production `train_svc.py` flow but pulls data
from the local SQLite master table instead of Snowflake. Use this to
verify the retrain end-to-end before plugging in real Snowflake creds.

Usage:
    1. Run `python scripts/local_dryrun.py` to build data/local_master.db
    2. Run `python scripts/local_retrain.py` to train + log to mlflow.db
    3. Run `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db`
       and open http://localhost:5000
"""
from __future__ import annotations

import logging
import os
import sqlite3
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("local_retrain")


def load_local_data() -> tuple[pd.DataFrame, pd.Series]:
    """Pull MASTER_CHURN_FEATURES from the local SQLite master table."""
    db_path = ROOT / "data" / "local_master.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run `python scripts/local_dryrun.py` first."
        )
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM MASTER_CHURN_FEATURES", conn)
    conn.close()
    logger.info("Loaded %d rows from local master", len(df))

    # The local master table has the full raw schema (Country, Province,
    # City, etc.) because the dryrun builds from CSVs. The production
    # Snowflake master table is pre-filtered to the 13 modeling columns.
    # Drop the extras here to match the production schema exactly.
    KEEP = [
        "LOYALTY_NUMBER", "GENDER", "EDUCATION", "SALARY", "MARITAL_STATUS",
        "LOYALTY_CARD", "CLV", "ENROLLMENT_TYPE",
        "LIFETIME_FLIGHTS", "LIFETIME_DISTANCE",
        "LIFETIME_POINTS_EARNED", "LIFETIME_POINTS_REDEEMED",
        "CHURN_FLAG",
    ]
    df = df[[c for c in KEEP if c in df.columns]]
    logger.info("Filtered to %d columns: %s", len(df.columns), list(df.columns))

    # Apply the same cleaning logic as `pull_snowflake.fetch_and_clean_data`
    if "SALARY" in df.columns:
        valid = df["SALARY"] >= 0
        if (~valid).any():
            df.loc[~valid, "SALARY"] = pd.NA
        if df["SALARY"].isna().any():
            median = df["SALARY"].median()
            df["SALARY"] = df["SALARY"].fillna(median)

    DROP_COLS = ["LOYALTY_NUMBER"]
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    CATEGORICAL = ["GENDER", "EDUCATION", "MARITAL_STATUS",
                   "LOYALTY_CARD", "ENROLLMENT_TYPE"]
    df_enc = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)
    dummy_cols = [c for c in df_enc.columns
                  if any(c.startswith(f"{c}_") for c in CATEGORICAL)]
    df_enc[dummy_cols] = df_enc[dummy_cols].astype(int)

    feature_names = [c for c in df_enc.columns if c != "CHURN_FLAG"]
    X = df_enc[feature_names].copy()
    y = df_enc["CHURN_FLAG"].astype(int)
    return X, y, feature_names


def main() -> int:
    X, y, feature_names = load_local_data()
    logger.info("X shape=%s, churn rate=%.3f, features=%d",
                X.shape, y.mean(), len(feature_names))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", C=10.0, class_weight="balanced",
                    probability=True, random_state=42)),
    ])

    db_path = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Airline_Churn_Production")

    with mlflow.start_run(run_name="SVC_Champion_Local") as run:
        params = {"kernel": "linear", "C": 10.0,
                  "class_weight": "balanced", "feature_count": len(feature_names)}
        mlflow.log_params(params)

        pipeline.fit(Xtr, ytr)
        pred = pipeline.predict(Xte)
        proba = pipeline.predict_proba(Xte)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(yte, pred)),
            "precision": float(precision_score(yte, pred, zero_division=0)),
            "recall": float(recall_score(yte, pred, zero_division=0)),
            "f1_score": float(f1_score(yte, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(yte, proba)),
        }
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        metrics.update({"true_negatives": float(tn), "false_positives": float(fp),
                        "false_negatives": float(fn), "true_positives": float(tp)})
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            pipeline, artifact_path="tuned_model",
            input_example=Xte.iloc[:2],
        )
        mlflow.log_text("\n".join(feature_names), "feature_names.txt")

        logger.info("=" * 60)
        logger.info("LOCAL CHAMPION TRAINED")
        logger.info("=" * 60)
        for k, v in metrics.items():
            logger.info("  %s = %.3f", k, v)
        logger.info("Run ID: %s", run.info.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
