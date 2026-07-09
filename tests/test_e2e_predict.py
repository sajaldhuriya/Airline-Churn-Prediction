"""
End-to-end test: Flask /predict -> FastAPI /predict.

This is the most important integration test in the suite. It exercises
the full path a manager clicks through in the browser:

  Browser POST /predict
      -> Flask parses form
      -> Flask POSTs JSON to FastAPI
      -> FastAPI runs the model
      -> Flask renders the result in the template

If any of those four hops break, this test breaks.

Strategy:
  1. Stand up a real uvicorn server in a subprocess running a stub FastAPI
     (the real one requires MLflow + a trained champion; this stub mirrors
     the contract so we can run hermetically).
  2. Use Flask's test client to POST /predict with form data.
  3. Patch `FASTAPI_URL` in the Flask module to point at our test server.

The test is hermetic: no Snowflake, no MLflow, no live model.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Form data shape — matches predict.html input names exactly. Mirrors the
# v2.0.0 API schema: 4 engagement numerics + 15 dummies. No SALARY, no CLV.
# Each categorical is "exactly one of N" by default value, and the test
# flips individual flags to exercise the API's exclusivity validators.
RETAINED_FORM = {
    "flights":           "28",
    "distance":          "30948",
    "points_earned":     "46422",
    "points_redeemed":   "1024",
    "gender_female":     "1", "gender_male":   "0",
    "edu_bachelor":      "0", "edu_college":   "0",
    "edu_doctor":        "0", "edu_highschool":"0", "edu_master": "1",
    "marital_divorced":  "0", "marital_married":"1", "marital_single":"0",
    "card_aurora":       "0", "card_nova":     "0", "card_star":   "1",
    "enroll_2018_promo": "0", "enroll_standard":"1",
}

CHURNER_FORM = dict(RETAINED_FORM)
CHURNER_FORM.update({
    "flights": "1", "distance": "200", "points_earned": "50", "points_redeemed": "0",
    "card_nova": "1", "card_star": "0",
})


# ---------------------------------------------------------------------------
# Stub FastAPI app written to a temp file and launched in a subprocess.
# Mirrors the real API contract: POST /predict -> {churn_prediction, probability}.
# ---------------------------------------------------------------------------
STUB_APP_SOURCE = '''
"""Stub FastAPI app used only by tests/test_e2e_predict.py.

The real src.api.app loads a model from MLflow, which makes end-to-end
testing brittle. This stub reproduces the response contract and is
launched in a subprocess by pytest. Schema mirrors v2.0.0: 4 numerics
+ 15 dummies, with one-of-N exclusivity validators.
"""
import math
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

app = FastAPI()


class CustomerData(BaseModel):
    LIFETIME_FLIGHTS: float = Field(..., ge=0)
    LIFETIME_DISTANCE: float = Field(..., ge=0)
    LIFETIME_POINTS_EARNED: float = Field(..., ge=0)
    LIFETIME_POINTS_REDEEMED: float = Field(..., ge=0)
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
    def _edu_exclusive(self):
        s = (self.EDUCATION_Bachelor + self.EDUCATION_College
             + self.EDUCATION_Doctor + self.EDUCATION_High_School_or_Below
             + self.EDUCATION_Master)
        if s != 1:
            raise ValueError("EDUCATION must select exactly one level")
        return self

    @model_validator(mode="after")
    def _gender_exclusive(self):
        if self.GENDER_Female + self.GENDER_Male != 1:
            raise ValueError("GENDER must select exactly one")
        return self

    @model_validator(mode="after")
    def _marital_exclusive(self):
        if (self.MARITAL_STATUS_Divorced + self.MARITAL_STATUS_Married
                + self.MARITAL_STATUS_Single) != 1:
            raise ValueError("MARITAL_STATUS must select exactly one")
        return self

    @model_validator(mode="after")
    def _card_exclusive(self):
        if (self.LOYALTY_CARD_Aurora + self.LOYALTY_CARD_Nova
                + self.LOYALTY_CARD_Star) != 1:
            raise ValueError("LOYALTY_CARD must select exactly one")
        return self

    @model_validator(mode="after")
    def _enroll_exclusive(self):
        if (self.ENROLLMENT_TYPE_2018_Promotion
                + self.ENROLLMENT_TYPE_Standard) != 1:
            raise ValueError("ENROLLMENT_TYPE must select exactly one")
        return self


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: CustomerData):
    # Hand-tuned score so the test fixtures flip labels correctly:
    #   RETAINED_FORM (Star, 28 flights) -> NO   (proba < 0.5)
    #   CHURNER_FORM  (Nova, 1 flight)   -> YES  (proba >= 0.5)
    # The score is dominated by the two card flags and lifetime flights,
    # with distance/points as smaller adjustments. Coefficients were
    # chosen by trial so the synthetic fixtures land on opposite sides
    # of 0.5 (verified in test_e2e_predict).
    score = (0.30
             - 0.15 * math.log1p(c.LIFETIME_FLIGHTS)
             - 0.02 * math.log1p(c.LIFETIME_DISTANCE)
             - 0.01 * math.log1p(c.LIFETIME_POINTS_EARNED)
             - 0.05 * math.log1p(c.LIFETIME_POINTS_REDEEMED)
             + 1.20 * c.LOYALTY_CARD_Nova
             - 1.30 * c.LOYALTY_CARD_Star
             + 0.20 * c.ENROLLMENT_TYPE_2018_Promotion
             + 0.01 * c.MARITAL_STATUS_Married
             + 0.01 * c.GENDER_Male)
    proba = 1.0 / (1.0 + math.exp(-score))
    label = "YES" if proba >= 0.5 else "NO"
    return {"churn_prediction": label,
            "probability": round(proba, 3),
            "served_by": "stub_fastapi_for_e2e_test"}
'''


@pytest.fixture(scope="module")
def fastapi_server():
    """Start a real uvicorn server in a subprocess; tear down at end of module."""
    port = _free_port()
    stub_path = ROOT / "tests" / "_stub_fastapi_for_e2e.py"
    stub_path.write_text(STUB_APP_SOURCE, encoding="utf-8")

    env = os.environ.copy()
    # Add the tests/ dir to PYTHONPATH so uvicorn can find the stub module.
    env["PYTHONPATH"] = str(ROOT / "tests") + os.pathsep + str(ROOT) \
        + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "_stub_fastapi_for_e2e:app",
         "--host", "127.0.0.1", "--port", str(port),
         "--log-level", "warning"],
        cwd=str(ROOT / "tests"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait up to 15s for /health to respond.
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            r = requests.get(base + "/health", timeout=0.5)
            if r.status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.1)
    else:
        proc.terminate()
        stderr_out = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        pytest.fail(f"Stub FastAPI did not become healthy in 15s.\n{stderr_out}")

    yield base

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        stub_path.unlink()
    except OSError:
        pass


@pytest.fixture
def flask_client(fastapi_server, monkeypatch):
    """A Flask test client pointed at our test FastAPI server."""
    monkeypatch.setenv("FASTAPI_URL", f"{fastapi_server}/predict")
    monkeypatch.setenv("FASTAPI_HEALTH_URL", f"{fastapi_server}/health")

    # Force a fresh import so module-level env reads pick up the new URL.
    for mod_name in [name for name in list(sys.modules)
                     if name.startswith("business_strategy.flask_dashboard")]:
        del sys.modules[mod_name]

    from business_strategy.flask_dashboard.app import app
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# The actual tests
# ---------------------------------------------------------------------------
class TestE2EPredict:
    def test_retained_user_renders_no_churn(self, flask_client):
        r = flask_client.post("/predict", data=RETAINED_FORM,
                              follow_redirects=True)
        assert r.status_code == 200, r.data[:500]
        # The result is rendered into the template. The stub returns
        # "churn_prediction": "NO" for an engaged Star-card holder.
        assert b"NO" in r.data

    def test_churner_renders_yes(self, flask_client):
        r = flask_client.post("/predict", data=CHURNER_FORM,
                              follow_redirects=True)
        assert r.status_code == 200, r.data[:500]
        assert b"YES" in r.data

    def test_form_validation_error_does_not_crash(self, flask_client):
        # Missing a required field -> Flask raises ValueError -> page
        # renders the error gracefully (status 200, no traceback).
        bad = dict(RETAINED_FORM)
        bad.pop("flights")
        r = flask_client.post("/predict", data=bad, follow_redirects=True)
        assert r.status_code == 200
        assert b"Invalid form input" in r.data

    def test_dual_education_rejected_by_api(self, flask_client, fastapi_server):
        # Form-side lets it through (each flag is a separate checkbox), but
        # the stub FastAPI's Pydantic validator rejects multi-valued
        # education with 422. Flask should surface that as an error message.
        bad = dict(RETAINED_FORM)
        bad["edu_college"] = "1"
        bad["edu_master"] = "1"
        r = flask_client.post("/predict", data=bad, follow_redirects=True)
        assert r.status_code == 200
        # Flask catches the non-200 from FastAPI and renders the error text.
        assert b"API engine returned 422" in r.data

    def test_dual_gender_rejected_by_api(self, flask_client, fastapi_server):
        bad = dict(RETAINED_FORM)
        bad["gender_female"] = "1"
        bad["gender_male"] = "1"
        r = flask_client.post("/predict", data=bad, follow_redirects=True)
        assert r.status_code == 200
        assert b"API engine returned 422" in r.data

    def test_no_card_rejected_by_api(self, flask_client, fastapi_server):
        bad = dict(RETAINED_FORM)
        bad["card_star"] = "0"
        # Star, Nova, Aurora are all 0 -> card validator fails
        bad["card_nova"] = "0"
        bad["card_aurora"] = "0"
        r = flask_client.post("/predict", data=bad, follow_redirects=True)
        assert r.status_code == 200
        assert b"API engine returned 422" in r.data

    def test_fastapi_down_surfaces_clean_error(self, monkeypatch):
        """If the FastAPI URL points at a dead port, the dashboard should
        show 'FastAPI engine unreachable' — not a 500 or stack trace."""
        # Point at a port nothing is listening on
        monkeypatch.setenv("FASTAPI_URL", f"http://127.0.0.1:{_free_port()}/predict")

        for mod_name in [name for name in list(sys.modules)
                         if name.startswith("business_strategy.flask_dashboard")]:
            del sys.modules[mod_name]
        from business_strategy.flask_dashboard.app import app
        app.config["TESTING"] = True

        r = app.test_client().post("/predict", data=RETAINED_FORM,
                                   follow_redirects=True)
        assert r.status_code == 200
        assert b"FastAPI engine unreachable" in r.data
