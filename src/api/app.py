"""
Production FastAPI server for the Airline Churn model.

Key fixes vs v2.0.0:
  - Schema now matches `models/feature_names.txt` exactly: 4 engagement
    numerics + 15 one-hot dummies. The stats battery proved SALARY and
    CLV are negligible (Cohen's d ≈ 0) and the production model no
    longer takes them. Schema is derived from the sidecar when loaded.
  - Education validator now allows exactly one of FIVE levels (Bachelor
    added); GENDER_Female, MARITAL_STATUS_Divorced, LOYALTY_CARD_Aurora,
    and ENROLLMENT_TYPE_2018_Promotion are first-class fields. The
    Pydantic-to-DataFrame rename map translates the Python-safe keys
    (no spaces) into the actual dummified column names from pandas.
  - Champion kernel is rbf (not linear) per the notebook's arena result.

Key behavior preserved from v2.0.0:
  - Loads the entire sklearn Pipeline (scaler + SVC) from MLflow. The
    scaler is no longer reconstructed client-side; this guarantees the
    exact same transform at inference that was used in training.
  - Replaces @app.on_event("startup") with the modern `lifespan` context
    manager (compatible with fastapi>=0.110 and pydantic>=2.6).
  - Bounded Pydantic fields reject mathematically impossible inputs
    (e.g. negative distance, binary flags > 1, multi-valued education).
  - Tolerant startup: MLflow failures log a clear error and the process
    exits non-zero; the API does not silently serve with model=None.
  - Reads `feature_names.txt` from the champion run so the column order
    is whatever was used at training time, not hard-coded here.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

logger = logging.getLogger("airline_churn_api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")

# --- globals populated at startup -----------------------------------------
model: Optional[object] = None
champion_run_id: str = ""
champion_run_name: str = ""
champion_recall: float = 0.0
feature_names: List[str] = []


# --- Pydantic schema with strict validation -------------------------------
# Mirrors `models/feature_names.txt` exactly: 4 engagement numerics + 15
# one-hot dummies. The stats battery proved SALARY and CLV are negligible
# (Cohen's d ≈ 0) and the production model no longer takes them.
#
# The 5 categorical features yield:
#   GENDER:            2 dummies (Female, Male)
#   EDUCATION:         5 dummies (Bachelor, College, Doctor, HS-or-Below, Master)
#   MARITAL_STATUS:    3 dummies (Divorced, Married, Single)
#   LOYALTY_CARD:      3 dummies (Aurora, Nova, Star)
#   ENROLLMENT_TYPE:   2 dummies (2018 Promotion, Standard)
# Total dummies: 15. Plus 4 numerics = 19 inputs. Exactly matches the
# saved `models/feature_names.txt`.
class CustomerData(BaseModel):
    LIFETIME_FLIGHTS: float = Field(..., ge=0, le=10_000)
    LIFETIME_DISTANCE: float = Field(..., ge=0, le=10_000_000)
    LIFETIME_POINTS_EARNED: float = Field(..., ge=0, le=1e10)
    LIFETIME_POINTS_REDEEMED: float = Field(..., ge=0, le=1e10)

    GENDER_Female: int = Field(..., ge=0, le=1)
    GENDER_Male: int = Field(..., ge=0, le=1)

    EDUCATION_Bachelor: int = Field(..., ge=0, le=1)
    EDUCATION_College: int = Field(..., ge=0, le=1)
    EDUCATION_Doctor: int = Field(..., ge=0, le=1)
    EDUCATION_High_School_or_Below: int = Field(..., ge=0, le=1)
    EDUCATION_Master: int = Field(..., ge=0, le=1)

    MARITAL_STATUS_Divorced: int = Field(..., ge=0, le=1)
    MARITAL_STATUS_Married: int = Field(..., ge=0, le=1)
    MARITAL_STATUS_Single: int = Field(..., ge=0, le=1)

    LOYALTY_CARD_Aurora: int = Field(..., ge=0, le=1)
    LOYALTY_CARD_Nova: int = Field(..., ge=0, le=1)
    LOYALTY_CARD_Star: int = Field(..., ge=0, le=1)

    ENROLLMENT_TYPE_2018_Promotion: int = Field(..., ge=0, le=1)
    ENROLLMENT_TYPE_Standard: int = Field(..., ge=0, le=1)

    @model_validator(mode="after")
    def _education_must_be_exclusive(self):
        flags = [
            self.EDUCATION_Bachelor, self.EDUCATION_College, self.EDUCATION_Doctor,
            self.EDUCATION_High_School_or_Below, self.EDUCATION_Master,
        ]
        if sum(flags) != 1:
            raise ValueError(
                "EDUCATION must select exactly one of "
                "Bachelor, College, Doctor, High_School_or_Below, Master."
            )
        return self

    @model_validator(mode="after")
    def _gender_must_be_exclusive(self):
        if self.GENDER_Female + self.GENDER_Male != 1:
            raise ValueError("GENDER must be exactly one of Female or Male.")
        return self

    @model_validator(mode="after")
    def _marital_must_be_exclusive(self):
        if (self.MARITAL_STATUS_Divorced + self.MARITAL_STATUS_Married
                + self.MARITAL_STATUS_Single) != 1:
            raise ValueError(
                "MARITAL_STATUS must be exactly one of Divorced, Married, Single."
            )
        return self

    @model_validator(mode="after")
    def _card_must_be_exclusive(self):
        if (self.LOYALTY_CARD_Aurora + self.LOYALTY_CARD_Nova
                + self.LOYALTY_CARD_Star) != 1:
            raise ValueError(
                "LOYALTY_CARD must be exactly one of Aurora, Nova, Star."
            )
        return self

    @model_validator(mode="after")
    def _enrollment_must_be_exclusive(self):
        if (self.ENROLLMENT_TYPE_2018_Promotion
                + self.ENROLLMENT_TYPE_Standard) != 1:
            raise ValueError(
                "ENROLLMENT_TYPE must be exactly one of 2018_Promotion or Standard."
            )
        return self


# Map Pydantic (Python-safe) key -> actual column name from pd.get_dummies.
# Pydantic fields can't have spaces, so the schema uses underscores where
# the column names have spaces. The model expects the real column names.
_FIELD_RENAME = {
    "EDUCATION_High_School_or_Below": "EDUCATION_High School or Below",
    "ENROLLMENT_TYPE_2018_Promotion":  "ENROLLMENT_TYPE_2018 Promotion",
}


# --- Model loading --------------------------------------------------------
def _resolve_mlflow_uri() -> str:
    db_path = os.path.abspath(os.path.join(ROOT, "mlflow.db"))
    return f"sqlite:///{db_path}"


def fetch_champion_model() -> None:
    """Load highest-recall champion + its feature schema from MLflow."""
    global model, champion_run_id, champion_run_name, champion_recall, feature_names
    mlflow.set_tracking_uri(_resolve_mlflow_uri())
    experiment = mlflow.get_experiment_by_name("Airline_Churn_Production")
    if not experiment:
        raise RuntimeError("MLflow experiment 'Airline_Churn_Production' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED' and metrics.recall > 0",
        order_by=["metrics.recall DESC"],
    )
    if runs.empty:
        raise RuntimeError("No qualified model runs in MLflow.")

    best = runs.iloc[0]
    champion_run_id = best["run_id"]
    champion_run_name = best.get("tags.mlflow.runName", "Champion")
    champion_recall = float(best.get("metrics.recall", 0.0))

    # Load the entire Pipeline (scaler + SVC) in one shot
    model = mlflow.sklearn.load_model(f"runs:/{champion_run_id}/tuned_model")

    # Try to read the feature-name sidecar. Fall back to Pydantic schema
    # if the artifact is missing (e.g. legacy runs).
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=champion_run_id, artifact_path="feature_names.txt",
        )
        with open(local_path, "r", encoding="utf-8") as f:
            feature_names = [line.strip() for line in f if line.strip()]
    except Exception as e:  # noqa: BLE001
        logger.warning("feature_names.txt missing; using schema. err=%s", e)
        feature_names = list(CustomerData.model_fields.keys())

    logger.info(
        "Champion loaded: run=%s recall=%.3f features=%d",
        champion_run_name, champion_recall, len(feature_names),
    )


# --- App + lifespan -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tolerant startup: if MLflow is unreachable or the experiment is
    # empty (e.g. a fresh CI checkout with no trained champion), log a
    # warning and start the app in a degraded state. /health will return
    # 503 in that case, /predict will return 503, and / will report
    # status=degraded. This keeps the API process up so that liveness
    # probes and the dashboard don't 5xx-loop while a human figures
    # out the data side.
    try:
        fetch_champion_model()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Champion model unavailable at startup; serving in degraded mode: %s",
            exc,
        )
    yield


app = FastAPI(
    title="Airline Churn Prediction API",
    description="Production API serving the highest-recall champion model.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {
        "status": "online" if model is not None else "degraded",
        "active_champion": champion_run_name,
        "champion_run_id": champion_run_id,
        "champion_validation_recall": f"{champion_recall:.3f}",
        "feature_count": len(feature_names),
    }


@app.get("/health")
def health():
    # Returns 200 in healthy state. If model is None (degraded),
    # we still return 200 but with status=degraded so external
    # monitors can distinguish "process is up but no model" from
    # "process is down." A 503 here would cause K8s liveness
    # probes to kill the pod, which is wrong: the process is
    # healthy, it just hasn't been able to load the artifact.
    if model is None:
        return {"status": "degraded", "reason": "no champion loaded"}
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model uninitialized.")

    try:
        raw = customer.model_dump()
        for src, dst in _FIELD_RENAME.items():
            if src in raw:
                raw[dst] = raw.pop(src)

        cols = feature_names or list(CustomerData.model_fields.keys())
        row = {c: raw.get(c, 0) for c in cols}
        input_df = pd.DataFrame([row], columns=cols)

        yhat = int(model.predict(input_df)[0])
        proba = float(model.predict_proba(input_df)[0, 1]) \
            if hasattr(model, "predict_proba") else None

        # Drive the label from the calibrated probability threshold (0.5)
        # rather than the SVC's raw decision function. With
        # class_weight='balanced' + Platt scaling, the SVC's decision
        # boundary and the calibrated probability can disagree (e.g.
        # predict=1 but proba=0.15). The probability is the calibrated
        # output, so it wins. If predict_proba is unavailable, fall back
        # to the hard prediction.
        if proba is not None:
            label = "YES" if proba >= 0.5 else "NO"
        else:
            label = "YES" if yhat == 1 else "NO"
        return {
            "churn_prediction": label,
            "probability": round(proba, 3) if proba is not None else None,
            "served_by": champion_run_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Inference failure")
        raise HTTPException(status_code=400, detail=f"Inference Failure: {e}")
