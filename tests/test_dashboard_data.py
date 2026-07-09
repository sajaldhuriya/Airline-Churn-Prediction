"""
Hermetic tests for the dashboard data layer.

The dashboard data layer (`business_strategy/flask_dashboard/data_access.py`)
is the only place in the project that mixes:
  - feature engineering (joining the two CSVs)
  - KPI computation (churn rate, revenue at risk, BCG segments)
  - statistical testing (Welch, Mann-Whitney, Chi^2, Levene)
  - model introspection (loading the champion to get |coef| for drivers)

These tests run with the real local CSVs but mock the MLflow layer so
they don't require a trained model. The full E2E flow (loading the
champion) is exercised by `tests/test_e2e_predict.py`.

In CI (where the raw CSVs are gitignored), the `_ensure_test_csvs`
fixture synthesizes a small fake dataset and points the data layer at
it via module-level patches. This keeps the suite hermetic on a clean
GitHub Actions runner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from business_strategy.flask_dashboard import data_access  # noqa: E402


@pytest.fixture(autouse=True)
def _use_synthetic_csvs(synthetic_csvs):
    """Shared fixture from conftest.py: real CSVs in dev, synthetic in CI."""
    yield


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Existing fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset module-level caches so each test sees fresh state."""
    data_access._load_local_frame.cache_clear()
    data_access.get_model_drivers.cache_clear()
    data_access._get_per_segment_recall_uncached.cache_clear()
    yield
    data_access._load_local_frame.cache_clear()
    data_access.get_model_drivers.cache_clear()
    data_access._get_per_segment_recall_uncached.cache_clear()


# ---------------------------------------------------------------------------
# KPI shape
# ---------------------------------------------------------------------------
def test_get_kpis_returns_expected_keys():
    kpis = data_access.get_kpis()
    for key in ("total_passengers", "churn_rate", "revenue_at_risk",
                "avg_flights", "card_churn", "drivers"):
        assert key in kpis, f"missing key: {key}"


def test_get_kpis_card_churn_is_dict_with_labels_and_values():
    kpis = data_access.get_kpis()
    cc = kpis["card_churn"]
    assert isinstance(cc, dict)
    assert "labels" in cc and "values" in cc
    assert len(cc["labels"]) == len(cc["values"])
    assert all(v >= 0 for v in cc["values"])


def test_get_kpis_slicer_filters_total_passengers():
    base = data_access.get_kpis()["total_passengers"]
    star_only = data_access.get_kpis(card="Star")["total_passengers"]
    assert star_only <= base, "slicer should not grow the population"


# ---------------------------------------------------------------------------
# Drivers: model-driven with snapshot fallback
# ---------------------------------------------------------------------------
def test_get_model_drivers_snapshot_fallback(monkeypatch):
    """If MLflow is unavailable, drivers fall back to the snapshot dict."""
    monkeypatch.setattr(data_access, "_load_champion_coefficients", lambda: None)
    data_access.get_model_drivers.cache_clear()
    drivers = data_access.get_model_drivers()
    assert "labels" in drivers and "values" in drivers
    assert sum(drivers["values"]) > 0
    # Snapshot dict has these exact 4 labels
    assert set(drivers["labels"]).issuperset({"Low engagement", "Points hoarded"})


def test_get_model_drivers_sums_to_100():
    """Live drivers come back as percentages summing to 100."""
    drivers = data_access.get_model_drivers()
    assert sum(drivers["values"]) == 100, (
        f"driver percentages should sum to 100, got {drivers['values']}"
    )
    assert len(drivers["labels"]) == len(drivers["values"]) > 0


def test_get_model_drivers_cached():
    """Repeated calls hit the LRU cache, so we only touch MLflow once."""
    data_access.get_model_drivers()
    data_access.get_model_drivers()
    # Cache hit is the whole point; we just confirm we got the same object.
    assert data_access.get_model_drivers() is data_access.get_model_drivers()


def test_get_model_drivers_top_n_respected(monkeypatch):
    """When the model has many features, only the top _DRIVER_TOP_N show."""
    fake_feats = [f"feat_{i}" for i in range(16)]
    fake_mags = np.array([float(i) for i in range(16)])  # monotonic
    monkeypatch.setattr(
        data_access, "_load_champion_coefficients",
        lambda: (fake_feats, fake_mags),
    )
    data_access.get_model_drivers.cache_clear()
    drivers = data_access.get_model_drivers()
    assert len(drivers["labels"]) == data_access._DRIVER_TOP_N
    # Largest magnitude (feat_15) should be first
    assert "feat_15" in drivers["labels"][0]


def test_get_model_drivers_handles_zero_magnitudes(monkeypatch):
    """All-zero magnitudes shouldn't divide-by-zero; should fall back."""
    monkeypatch.setattr(
        data_access, "_load_champion_coefficients",
        lambda: (["a", "b"], np.array([0.0, 0.0])),
    )
    data_access.get_model_drivers.cache_clear()
    drivers = data_access.get_model_drivers()
    # Falls back to snapshot
    assert "labels" in drivers and "values" in drivers


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def test_bootstrap_churn_rate_ci_basic():
    flags = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    ci = data_access.bootstrap_churn_rate_ci(flags, n_resamples=200, seed=7)
    assert ci is not None
    assert ci["point"] == 0.5
    assert ci["lo"] <= 0.5 <= ci["hi"]
    assert ci["lo"] >= 0 and ci["hi"] <= 1


def test_bootstrap_churn_rate_ci_too_few_obs():
    assert data_access.bootstrap_churn_rate_ci(pd.Series([1])) is None
    assert data_access.bootstrap_churn_rate_ci(pd.Series([])) is None


def test_bootstrap_churn_rate_ci_seeded_reproducible():
    """Stakeholder demo: same data + same seed = same CI."""
    flags = pd.Series(np.random.default_rng(0).integers(0, 2, 500))
    a = data_access.bootstrap_churn_rate_ci(flags, n_resamples=200, seed=42)
    b = data_access.bootstrap_churn_rate_ci(flags, n_resamples=200, seed=42)
    assert a == b


def test_bootstrap_churn_rate_ci_narrows_with_n():
    """Wider sample should give a tighter CI than narrow sample."""
    rng = np.random.default_rng(0)
    big = pd.Series(rng.integers(0, 2, 5000))
    small = pd.Series(rng.integers(0, 2, 50))
    big_ci = data_access.bootstrap_churn_rate_ci(big, n_resamples=300, seed=1)
    small_ci = data_access.bootstrap_churn_rate_ci(small, n_resamples=300, seed=1)
    assert big_ci["half_width"] < small_ci["half_width"]


def test_get_kpis_includes_churn_rate_ci():
    kpis = data_access.get_kpis()
    assert "churn_rate_ci" in kpis
    ci = kpis["churn_rate_ci"]
    assert ci is not None
    for key in ("point", "lo", "hi", "half_width"):
        assert key in ci
    # CI should bracket the point estimate
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert 0 <= ci["lo"] <= ci["hi"] <= 1


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------
def test_get_hypothesis_tests_returns_four():
    """With real local data, we expect all four tests in the battery."""
    tests = data_access.get_hypothesis_tests()
    ids = [t["id"] for t in tests]
    assert "T-001" in ids, "Welch t-test missing"
    assert "T-002" in ids, "Mann-Whitney U missing"
    assert "T-003" in ids, "Chi-square / Cramer's V missing"
    assert "T-004" in ids, "Levene's test missing"


def test_get_hypothesis_tests_t001_has_variance_check():
    """The Welch t-test should carry a variance_check field for T-001."""
    tests = data_access.get_hypothesis_tests()
    t001 = next(t for t in tests if t["id"] == "T-001")
    assert "variance_check" in t001
    # The note should reference the variance ratio
    assert "ratio" in t001["variance_check"]


def test_get_hypothesis_tests_t003_has_cramers_v_in_insight():
    """The Chi-square test should embed Cramer's V in the insight text."""
    tests = data_access.get_hypothesis_tests()
    t003 = next(t for t in tests if t["id"] == "T-003")
    insight = t003["insight"].lower()
    if "cramér" in insight or "cramer" in insight:
        # If the test reached rejection, Cramer's V should be quoted
        if "fail to reject" not in t003["status"].lower():
            assert "v=" in insight or "v =" in insight or "v =" in insight or "v=" in insight
    # If the test failed to reject, the insight still mentions Cramer's V
    # in the "no meaningful difference" branch.
    assert "v=" in insight.replace(" ", "") or "cramer" in insight or "cramér" in insight


def test_get_hypothesis_tests_t004_is_levene():
    tests = data_access.get_hypothesis_tests()
    t004 = next(t for t in tests if t["id"] == "T-004")
    assert "Levene" in t004["test_type"]
    # The insight should mention the variance ratio
    assert "ratio" in t004["insight"].lower()


# ---------------------------------------------------------------------------
# Per-segment recall (notebook §7 surface on the dashboard)
# ---------------------------------------------------------------------------
class _MockPipeline:
    """Deterministic stand-in for the trained champion used by segment tests.

    Predicts churn whenever the customer has the LOYALTY_CARD_Star
    flag set OR the ENROLLMENT_TYPE_2018_Promotion flag set. The rest
    of the feature row is ignored. This is brittle on purpose: we
    want the test to be a faithful stand-in for an SVC that genuinely
    struggles on the 2018-Promotion cohort.
    """
    def predict(self, X):
        if hasattr(X, "columns"):
            star = X["LOYALTY_CARD_Star"].to_numpy() if "LOYALTY_CARD_Star" in X.columns else 0
            promo = (X["ENROLLMENT_TYPE_2018 Promotion"].to_numpy()
                     if "ENROLLMENT_TYPE_2018 Promotion" in X.columns else 0)
        else:
            star = X[:, X.columns.get_loc("LOYALTY_CARD_Star")] \
                if "LOYALTY_CARD_Star" in X.columns else 0
            promo = X[:, X.columns.get_loc("ENROLLMENT_TYPE_2018 Promotion")] \
                if "ENROLLMENT_TYPE_2018 Promotion" in X.columns else 0
        import numpy as _np
        return ((star + promo) > 0).astype(int)


def test_get_per_segment_recall_returns_rows(monkeypatch):
    """Live segments should return one row per level with at least
    _SEGMENT_MIN_N rows in the synthetic fixture."""
    fake_feats = [
        "LIFETIME_FLIGHTS", "LIFETIME_DISTANCE",
        "LIFETIME_POINTS_EARNED", "LIFETIME_POINTS_REDEEMED",
        "GENDER_Female", "GENDER_Male",
        "EDUCATION_Bachelor", "EDUCATION_College", "EDUCATION_Doctor",
        "EDUCATION_High School or Below", "EDUCATION_Master",
        "MARITAL_STATUS_Divorced", "MARITAL_STATUS_Married",
        "MARITAL_STATUS_Single",
        "LOYALTY_CARD_Aurora", "LOYALTY_CARD_Nova", "LOYALTY_CARD_Star",
        "ENROLLMENT_TYPE_2018 Promotion", "ENROLLMENT_TYPE_Standard",
    ]
    monkeypatch.setattr(
        data_access, "_load_champion_for_segments",
        lambda: (_MockPipeline(), fake_feats),
    )
    data_access._get_per_segment_recall_uncached.cache_clear()
    rows = data_access.get_per_segment_recall("LOYALTY_CARD")
    assert isinstance(rows, list)
    # Synthetic fixture has 3 LOYALTY_CARD levels, all with n>=30.
    # The mock predicts churn when Star=1, so Star recall will be
    # near 1.0; the other two will be near 0.0.
    assert len(rows) == 3
    # Weakest first (sorted ascending by recall)
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls)
    for r in rows:
        assert {"segment", "n", "churn_rate", "recall",
                "true_positives", "false_negatives"} <= set(r.keys())
        assert 0.0 <= r["recall"] <= 1.0
        assert r["n"] >= data_access._SEGMENT_MIN_N


def test_get_per_segment_recall_unknown_column(monkeypatch):
    """An unknown segment column should return [], not crash."""
    monkeypatch.setattr(data_access, "_load_champion_for_segments", lambda: None)
    data_access._get_per_segment_recall_uncached.cache_clear()
    assert data_access.get_per_segment_recall("FOO_BAR") == []


def test_get_per_segment_recall_no_champion(monkeypatch):
    """When MLflow is unavailable, the function returns an empty list."""
    monkeypatch.setattr(data_access, "_load_champion_for_segments", lambda: None)
    data_access._get_per_segment_recall_uncached.cache_clear()
    for col in ("LOYALTY_CARD", "EDUCATION", "GENDER",
                "MARITAL_STATUS", "ENROLLMENT_TYPE"):
        assert data_access.get_per_segment_recall(col) == []


def test_get_per_segment_recall_inference_failure(monkeypatch):
    """If the pipeline raises, we return [] rather than 500 the page."""
    class _Boom:
        def predict(self, X):
            raise RuntimeError("simulated inference failure")
    monkeypatch.setattr(
        data_access, "_load_champion_for_segments",
        lambda: (_Boom(), []),
    )
    data_access._get_per_segment_recall_uncached.cache_clear()
    assert data_access.get_per_segment_recall("LOYALTY_CARD") == []


# ---------------------------------------------------------------------------
# BCG
# ---------------------------------------------------------------------------
def test_get_bcg_segments_shape():
    bcg = data_access.get_bcg_segments()
    assert set(bcg.keys()) == {"stars", "cash_cows", "question_marks", "dogs"}
    # The four segments should sum to ~100 (rounding tolerance)
    total = sum(bcg.values())
    assert 95 <= total <= 105, f"BCG percentages off: {bcg}"


def test_get_bcg_segments_slicer_changes_segments():
    base = data_access.get_bcg_segments()
    star = data_access.get_bcg_segments(card="Star")
    # Slicer should change at least one segment
    assert base != star
