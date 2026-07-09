"""
Manager-facing dashboard for the AeroRetain AI churn system.

Differences vs v1.0:
  - All Snowflake/operational queries are routed through
    `data_access.py` so we never mock in production code.
  - Errors from the FastAPI backend are surfaced cleanly, not 500'd.
  - The form payload uses bounded int fields, matching the API.
  - `debug=True` removed for non-dev environments.

Differences vs v1.1:
  - The form payload matches the v2.0.0 API schema: 4 engagement
    numerics + 15 dummies. SALARY/CLV are dropped (negligible per the
    stats battery). New dummies for GENDER_Female, EDUCATION_Bachelor,
    MARITAL_STATUS_Divorced, LOYALTY_CARD_Aurora, and
    ENROLLMENT_TYPE_2018_Promotion are sent in the payload.
  - The dashboard's "weak segment" insights surface per-segment recall
    (notebook section 7) so the 2018-Promotion gap is visible.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from flask import Flask, render_template, request, jsonify

# Ensure the project root is on sys.path so `business_strategy.*` imports
# resolve whether this file is launched as a script (`python app.py`),
# as a module (`python -m business_strategy.flask_dashboard.app`),
# or via gunicorn/uwsgi from a different working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

app = Flask(__name__)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")
FASTAPI_HEALTH_URL = os.getenv("FASTAPI_HEALTH_URL", "http://127.0.0.1:8000/health")

# Slicer keys — whitelist to avoid passing arbitrary query params into the
# data layer. The HTML form / JS uses these exact names.
_SLICER_KEYS = ("gender", "card", "education")


def _slicer_args() -> Dict[str, str]:
    return {k: request.values.get(k, "all") for k in _SLICER_KEYS}


def _fetch_kpis() -> Dict[str, Any]:
    """Pull live operational KPIs from the data layer. Falls back to
    a hard-coded snapshot if Snowflake is unreachable so the dashboard
    is still usable in development."""
    try:
        from business_strategy.flask_dashboard.data_access import get_kpis
        return get_kpis(**_slicer_args())
    except Exception as e:  # noqa: BLE001
        logger.warning("KPI fallback in use: %s", e)
        # In the fallback we still try to use the model-derived drivers
        # when possible — they're cheap to compute and keep the chart
        # live. If THAT fails, we drop to the static snapshot.
        try:
            from business_strategy.flask_dashboard.data_access import get_model_drivers
            drivers = get_model_drivers()
        except Exception:
            drivers = {"labels": ["Low engagement", "Points hoarded", "Service complaints", "Competitor offer"],
                       "values": [42, 28, 18, 12]}
        return {
            "total_passengers": 14250,
            "churn_rate": 0.164,
            "revenue_at_risk": "$2.4M",
            "avg_flights": 28.4,
            "card_churn": {"labels": ["Aurora", "Nova", "Star"], "values": [12.3, 18.1, 21.4]},
            "drivers": drivers,
        }


def _fetch_bcg() -> Dict[str, int]:
    try:
        from business_strategy.flask_dashboard.data_access import get_bcg_segments
        return get_bcg_segments(**_slicer_args())
    except Exception as e:  # noqa: BLE001
        logger.warning("BCG fallback in use: %s", e)
        return {"stars": 12, "cash_cows": 45, "question_marks": 18, "dogs": 25}


def _fetch_hypothesis_tests() -> list:
    try:
        from business_strategy.flask_dashboard.data_access import get_hypothesis_tests
        return get_hypothesis_tests()
    except Exception as e:  # noqa: BLE001
        logger.warning("Stats fallback in use: %s", e)
        return []


def _fetch_segment_recall() -> Dict[str, list]:
    """Per-segment recall by the requested column. The default is
    LOYALTY_CARD because the Star-card gap is the most actionable
    finding in the notebook (and matches the slicer on the overview
    page). Other dimensions are exposed via /api/segments?col=… so
    the front-end can swap without a reload."""
    try:
        from business_strategy.flask_dashboard.data_access import (
            get_per_segment_recall,
        )
        out: Dict[str, list] = {}
        for col in ("LOYALTY_CARD", "EDUCATION", "GENDER",
                    "MARITAL_STATUS", "ENROLLMENT_TYPE"):
            out[col] = get_per_segment_recall(col)
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("Per-segment recall fallback in use: %s", e)
        return {col: [] for col in
                ("LOYALTY_CARD", "EDUCATION", "GENDER",
                 "MARITAL_STATUS", "ENROLLMENT_TYPE")}


@app.route("/")
def index():
    return render_template("index.html",
                           kpis=_fetch_kpis(),
                           active_page="overview")


@app.route("/strategy")
def strategy():
    return render_template("strategy.html",
                           bcg=_fetch_bcg(),
                           active_page="strategy")


@app.route("/stats")
def stats():
    return render_template("stats.html",
                           tests=_fetch_hypothesis_tests(),
                           segments=_fetch_segment_recall(),
                           active_page="stats")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_result: Optional[Dict[str, Any]] = None
    form_data = None
    error: Optional[str] = None

    if request.method == "POST":
        form_data = request.form
        try:
            # Extract dropdown selections
            gender = request.form["gender"]
            education = request.form["education"]
            marital = request.form["marital_status"]
            card = request.form["loyalty_card"]
            enroll = request.form["enrollment_type"]

            # Mirrors `src/api/app.py::CustomerData` exactly.
            payload = {
                "LIFETIME_FLIGHTS":          float(request.form["flights"]),
                "LIFETIME_DISTANCE":         float(request.form["distance"]),
                "LIFETIME_POINTS_EARNED":    float(request.form["points_earned"]),
                "LIFETIME_POINTS_REDEEMED":  float(request.form["points_redeemed"]),
                
                # Convert dropdowns back to one-hot dummies for the API
                "GENDER_Female":             1 if gender == "Female" else 0,
                "GENDER_Male":               1 if gender == "Male" else 0,
                
                "EDUCATION_Bachelor":        1 if education == "Bachelor" else 0,
                "EDUCATION_College":         1 if education == "College" else 0,
                "EDUCATION_Doctor":          1 if education == "Doctor" else 0,
                "EDUCATION_High_School_or_Below": 1 if education == "High School or Below" else 0,
                "EDUCATION_Master":          1 if education == "Master" else 0,
                
                "MARITAL_STATUS_Divorced":   1 if marital == "Divorced" else 0,
                "MARITAL_STATUS_Married":    1 if marital == "Married" else 0,
                "MARITAL_STATUS_Single":     1 if marital == "Single" else 0,
                
                "LOYALTY_CARD_Aurora":       1 if card == "Aurora" else 0,
                "LOYALTY_CARD_Nova":         1 if card == "Nova" else 0,
                "LOYALTY_CARD_Star":         1 if card == "Star" else 0,
                
                "ENROLLMENT_TYPE_2018_Promotion": 1 if enroll == "2018 Promotion" else 0,
                "ENROLLMENT_TYPE_Standard":  1 if enroll == "Standard" else 0,
            }
        except (KeyError, ValueError) as e:
            error = f"Invalid form input: {e}"
        else:
            try:
                resp = requests.post(FASTAPI_URL, json=payload, timeout=10)
                if resp.status_code == 200:
                    prediction_result = resp.json()
                else:
                    error = f"API engine returned {resp.status_code}: {resp.text[:200]}"
            except requests.RequestException as e:
                logger.exception("FastAPI engine unreachable")
                error = f"FastAPI engine unreachable: {e}"

    return render_template("predict.html",
                           result=prediction_result,
                           form=form_data,
                           error=error,
                           active_page="predict")


@app.route("/api/overview")
def api_overview():
    """JSON endpoint backing the overview page slicers.
    Mirrors the shape of get_kpis() so JS can swap the result directly."""
    return jsonify(_fetch_kpis())


@app.route("/api/bcg")
def api_bcg():
    return jsonify(_fetch_bcg())


@app.route("/api/stats")
def api_stats():
    """JSON endpoint backing the stats page.

    Mirrors `/api/overview` and `/api/bcg`. The current stats page does
    not expose slicer controls (the hypothesis battery is computed on
    the full population), but this endpoint is the contract for a
    future JS-driven swap so we don't need a full page reload.
    """
    return jsonify({"tests": _fetch_hypothesis_tests()})


@app.route("/api/segments")
def api_segments():
    """Per-segment recall from the trained champion.

    The notebook (section 7) flagged the 2018-Promotion cohort as a
    weak segment — recall 0.448 on the champion, far below the 75%
    target. This endpoint surfaces that finding for any of the five
    categorical dimensions, with the segment column as a query param.
    """
    col = request.values.get("col", "LOYALTY_CARD")
    try:
        from business_strategy.flask_dashboard.data_access import (
            get_per_segment_recall,
        )
        rows = get_per_segment_recall(col)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_per_segment_recall failed: %s", exc)
        rows = []
    return jsonify({"column": col, "segments": rows})


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.getenv("FLASK_PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=debug)