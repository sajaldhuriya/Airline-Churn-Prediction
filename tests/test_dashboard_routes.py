"""
Hermetic tests for the Flask dashboard routes.

The /api/* endpoints are the seam between the dashboard front-end and
the live data layer. This test exercises the route handlers directly
with Flask's test client, so we don't need a real HTTP server, a real
MLflow, or a real champion model.

Why a separate file from test_dashboard_data.py:
  - test_dashboard_data.py exercises the data layer in isolation
    (no Flask context). It's faster and more focused.
  - This file exercises the route surface area: 200 status, JSON shape,
    fallback behavior. The test client is the right tool for that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from business_strategy.flask_dashboard import data_access  # noqa: E402
from business_strategy.flask_dashboard.app import app  # noqa: E402


@pytest.fixture(autouse=True)
def _use_synthetic_csvs(synthetic_csvs):
    """Shared fixture from conftest.py: real CSVs in dev, synthetic in CI."""
    yield


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Pages (server-rendered, smoke test)
# ---------------------------------------------------------------------------
def test_index_renders(client):
    """The overview page should always render — 200 + 'AeroRetain' in body."""
    resp = client.get("/")
    assert resp.status_code == 200
    # The title is in <title> on the base template
    assert b"Airline Churn" in resp.data or b"AeroRetain" in resp.data


def test_stats_page_renders(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    # The hypothesis battery should show at least one test ID
    assert b"T-001" in resp.data


def test_strategy_page_renders(client):
    resp = client.get("/strategy")
    assert resp.status_code == 200


def test_predict_page_renders(client):
    resp = client.get("/predict")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/overview — JSON shape
# ---------------------------------------------------------------------------
def test_api_overview_returns_kpis(client):
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    assert resp.is_json
    data = resp.get_json()
    for key in ("total_passengers", "churn_rate", "churn_rate_ci",
                "revenue_at_risk", "avg_flights", "card_churn", "drivers"):
        assert key in data, f"missing key: {key}"


def test_api_overview_slicer_filters(client):
    base = client.get("/api/overview").get_json()
    star = client.get("/api/overview?card=Star").get_json()
    assert star["total_passengers"] <= base["total_passengers"]


# ---------------------------------------------------------------------------
# /api/bcg — JSON shape
# ---------------------------------------------------------------------------
def test_api_bcg_returns_segments(client):
    resp = client.get("/api/bcg")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data.keys()) == {"stars", "cash_cows", "question_marks", "dogs"}


# ---------------------------------------------------------------------------
# /api/stats — the new endpoint
# ---------------------------------------------------------------------------
def test_api_stats_returns_tests(client):
    """The new /api/stats endpoint should return the hypothesis battery."""
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    assert resp.is_json
    data = resp.get_json()
    assert "tests" in data
    assert isinstance(data["tests"], list)
    assert len(data["tests"]) >= 4, "expected 4+ tests in the battery"


def test_api_stats_test_shape(client):
    """Each test entry should have the keys the front-end expects."""
    resp = client.get("/api/stats")
    data = resp.get_json()
    for test in data["tests"]:
        for key in ("id", "metric", "status", "test_type",
                    "p_value", "null_hypothesis", "insight"):
            assert key in test, f"test {test.get('id', '?')} missing {key}"


def test_api_stats_includes_levene_and_t001_variance_check(client):
    """The new T-004 Levene test should be in the response, and T-001
    should carry its variance_check field."""
    data = client.get("/api/stats").get_json()
    ids = [t["id"] for t in data["tests"]]
    assert "T-004" in ids, "Levene (T-004) missing from /api/stats"
    t001 = next(t for t in data["tests"] if t["id"] == "T-001")
    assert "variance_check" in t001, "T-001 missing variance_check"
    assert "ratio" in t001["variance_check"]


# ---------------------------------------------------------------------------
# Slicer plumbing — when MLflow/data layer fails, fallback kicks in
# ---------------------------------------------------------------------------
def test_api_overview_falls_back_when_data_access_raises(client, monkeypatch):
    """If data_access.get_kpis() raises, the API should still return 200
    with the snapshot dict (so the dashboard never 500s)."""
    def _boom(**_):
        raise RuntimeError("simulated outage")
    monkeypatch.setattr(data_access, "get_kpis", _boom)
    resp = client.get("/api/overview")
    assert resp.status_code == 200
    data = resp.get_json()
    # Snapshot has the same shape as live, so the front-end doesn't care
    assert "churn_rate" in data
    assert "drivers" in data


def test_api_stats_returns_empty_list_when_data_access_raises(client, monkeypatch):
    def _boom():
        raise RuntimeError("simulated outage")
    monkeypatch.setattr(data_access, "get_hypothesis_tests", _boom)
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {"tests": []}


# ---------------------------------------------------------------------------
# /api/segments — per-segment recall (notebook §7 surface on the dashboard)
# ---------------------------------------------------------------------------
def test_api_segments_returns_rows(client, monkeypatch):
    """The new /api/segments endpoint should return a row per level."""
    fake_rows = [
        {"segment": "2018 Promotion", "n": 200, "churn_rate": 0.12,
         "recall": 0.45, "true_positives": 11, "false_negatives": 13},
        {"segment": "Standard",       "n": 5000, "churn_rate": 0.12,
         "recall": 0.86, "true_positives": 514, "false_negatives": 84},
    ]
    monkeypatch.setattr(
        data_access, "get_per_segment_recall",
        lambda col="LOYALTY_CARD": fake_rows,
    )
    resp = client.get("/api/segments?col=ENROLLMENT_TYPE")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["column"] == "ENROLLMENT_TYPE"
    assert body["segments"] == fake_rows


def test_api_segments_falls_back_when_data_access_raises(client, monkeypatch):
    """If get_per_segment_recall raises, the API still returns 200 with []."""
    def _boom(col="LOYALTY_CARD"):
        raise RuntimeError("simulated outage")
    monkeypatch.setattr(data_access, "get_per_segment_recall", _boom)
    resp = client.get("/api/segments?col=LOYALTY_CARD")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["column"] == "LOYALTY_CARD"
    assert body["segments"] == []


def test_stats_page_renders_segments(client, monkeypatch):
    """The /stats page should render the per-segment recall tables."""
    fake_segments = {
        "LOYALTY_CARD": [
            {"segment": "Star", "n": 100, "churn_rate": 0.12,
             "recall": 0.80, "true_positives": 10, "false_negatives": 2},
        ],
        "ENROLLMENT_TYPE": [
            {"segment": "2018 Promotion", "n": 50, "churn_rate": 0.12,
             "recall": 0.45, "true_positives": 3, "false_negatives": 4},
        ],
        "EDUCATION": [], "GENDER": [], "MARITAL_STATUS": [],
    }
    monkeypatch.setattr(
        "business_strategy.flask_dashboard.app._fetch_segment_recall",
        lambda: fake_segments,
    )
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert b"Per-Segment Recall" in resp.data
    assert b"2018 Promotion" in resp.data
    assert b"Star" in resp.data
