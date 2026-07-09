"""
Hermetic tests for the drift detector.

Strategy:
  - Build small synthetic DataFrames where we KNOW the answer:
      * Identical distributions -> no drift on any feature.
      * Heavy shift (mean+5*sd) on one feature -> drift on that one.
      * New category in recent -> drift on that categorical.
      * Tiny shift (mean+0.01*sd) on numeric -> no drift (effect is
        negligible even if p < 0.05).
  - Assert on the report's `any_drift`, `drifted_features`, severity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import drift  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def baseline(rng):
    return pd.DataFrame({
        "FLOAT_A":   rng.normal(0, 1, 1000),
        "FLOAT_B":   rng.normal(10, 2, 1000),
        "CAT_X":     rng.choice(["a", "b", "c"], size=1000, p=[0.5, 0.3, 0.2]),
        "CAT_BIN":   rng.integers(0, 2, 1000),
    })


# ---------------------------------------------------------------------------
# No drift
# ---------------------------------------------------------------------------
def test_identical_distributions_no_drift(baseline, rng):
    """Same distribution in both frames -> no drift on any feature."""
    recent = baseline.sample(frac=1.0, random_state=rng.integers(0, 1_000_000)).reset_index(drop=True)
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    assert not report.any_drift, f"unexpected drift on: {report.drifted_features}"


# ---------------------------------------------------------------------------
# Heavy shift on a numeric -> drift
# ---------------------------------------------------------------------------
def test_heavy_numeric_shift_is_drift(baseline, rng):
    recent = baseline.copy()
    # Mean+5*sd shift on FLOAT_A: clearly different distribution
    recent["FLOAT_A"] = rng.normal(5.0, 1.0, 1000)
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    assert "FLOAT_A" in report.drifted_features
    assert "FLOAT_B" not in report.drifted_features
    # The drifted feature's effect size should be in 'medium' or 'large'
    a = next(f for f in report.features if f.feature == "FLOAT_A")
    assert a.severity in ("medium", "large"), f"unexpected severity: {a.severity}"


# ---------------------------------------------------------------------------
# New category in recent -> drift on that categorical
# ---------------------------------------------------------------------------
def test_new_category_is_drift(baseline, rng):
    recent = baseline.copy()
    # Replace 30% of CAT_X with a brand-new value present only in recent
    idx = rng.choice(len(recent), size=300, replace=False)
    recent.loc[idx, "CAT_X"] = "z"
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    assert "CAT_X" in report.drifted_features


def test_proportional_shift_is_drift(baseline, rng):
    """If the proportions change but the categories are the same, we
    should still detect it (the test is on the *distribution*, not the
    set of values)."""
    recent = baseline.copy()
    # Recent has heavily inverted proportions for CAT_X
    recent["CAT_X"] = rng.choice(["a", "b", "c"], size=1000, p=[0.1, 0.4, 0.5])
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    assert "CAT_X" in report.drifted_features


# ---------------------------------------------------------------------------
# Tiny shift -> no drift (effect is negligible)
# ---------------------------------------------------------------------------
def test_tiny_numeric_shift_not_drift(baseline, rng):
    """Mean+0.05*sd shift: p-value will be tiny on n=1000, but the
    effect size is negligible, so the report should not flag it.

    This is the n=16K statistical-power trap the stats battery warns
    about (T-001 / T-003 in data_access.py)."""
    recent = baseline.copy()
    recent["FLOAT_A"] = rng.normal(0.05, 1.0, 1000)  # mean+0.05*sd
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    assert "FLOAT_A" not in report.drifted_features, (
        "tiny shift should not be flagged as drift (negligible effect)"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_handles_missing_column_gracefully(baseline, rng):
    """If a column listed in numeric_cols/categorical_cols is missing
    from one of the frames, skip it with a warning rather than crash."""
    recent = baseline.drop(columns=["FLOAT_B"])
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    # FLOAT_A should still be checked
    a = next((f for f in report.features if f.feature == "FLOAT_A"), None)
    assert a is not None
    # FLOAT_B should be skipped (not in the report)
    assert "FLOAT_B" not in [f.feature for f in report.features]


def test_handles_nans_in_numeric(baseline, rng):
    """A handful of NaNs in numeric features should not crash the KS test."""
    baseline = baseline.copy()
    recent = baseline.copy()
    baseline.loc[baseline.index[:5], "FLOAT_A"] = np.nan
    recent.loc[recent.index[:5], "FLOAT_A"] = np.nan
    # Tiny shift on FLOAT_B so the report is otherwise clean
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    a = next(f for f in report.features if f.feature == "FLOAT_A")
    assert a.status == "stable"


def test_handles_small_sample():
    """If a column has <2 valid values in either frame, the test should
    not crash — it should just return a 'stable' result."""
    base = pd.DataFrame({"FLOAT_A": [1.0, np.nan, np.nan, np.nan]})
    rec  = pd.DataFrame({"FLOAT_A": [2.0, 2.0, 2.0, 2.0]})
    report = drift.detect_drift(
        base, rec,
        numeric_cols=["FLOAT_A"],
        categorical_cols=[],
    )
    assert len(report.features) == 1
    assert report.features[0].status == "stable"


# ---------------------------------------------------------------------------
# Report shape and write
# ---------------------------------------------------------------------------
def test_report_to_dict_shape(baseline, rng):
    recent = baseline.copy()
    recent["FLOAT_A"] = rng.normal(5.0, 1.0, 1000)
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    d = report.to_dict()
    assert d["baseline_n"] == 1000
    assert d["recent_n"]   == 1000
    assert d["alpha"]      == drift._ALPHA
    assert d["any_drift"]  is True
    assert "FLOAT_A" in d["drifted_features"]
    assert isinstance(d["features"], list)
    for f in d["features"]:
        for key in ("feature", "kind", "statistic", "p_value", "severity", "status"):
            assert key in f


def test_write_report(tmp_path, baseline, rng):
    recent = baseline.copy()
    out = tmp_path / "drift.json"
    report = drift.detect_drift(
        baseline, recent,
        numeric_cols=["FLOAT_A", "FLOAT_B", "CAT_BIN"],
        categorical_cols=["CAT_X"],
    )
    path = drift.write_report(report, str(out))
    assert path == str(out)
    with open(path) as f:
        data = json.load(f)
    assert "features" in data
    assert "any_drift" in data
