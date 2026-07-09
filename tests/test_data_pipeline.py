"""
Hermetic test of the Snowflake data pipeline.

Strategy: we don't fake a Snowflake connection. Instead we monkeypatch
`pull_snowflake._default_connect` to return a real SQLite connection
that serves a small dataframe. This exercises the real pandas code path
end-to-end without any external dependency.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import pull_snowflake  # noqa: E402

SAMPLE = pd.DataFrame({
    "LOYALTY_NUMBER": [1, 2, 3, 4, 5, 6, 7, 8],
    "GENDER":          ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
    "EDUCATION":       ["Bachelor", "Master", "College", "Doctor",
                        "High School or Below", "Bachelor", "Master", "College"],
    "SALARY":          [80000.0, -1000.0, None, 60000.0,
                        50000.0, 90000.0, 75000.0, 65000.0],  # -1000 = bad data
    "MARITAL_STATUS":  ["Married", "Single", "Divorced", "Married",
                        "Single", "Divorced", "Married", "Single"],
    "LOYALTY_CARD":    ["Star", "Nova", "Aurora", "Star",
                        "Nova", "Aurora", "Star", "Nova"],
    "CLV":             [5000.0, 7000.0, 9000.0, 3000.0,
                        4500.0, 8500.0, 6200.0, 7100.0],
    "ENROLLMENT_TYPE": ["Standard", "2018 Promotion", "Standard", "2018 Promotion",
                        "Standard", "2018 Promotion", "Standard", "2018 Promotion"],
    "LIFETIME_FLIGHTS":          [28, 11, 50, 35, 12, 60, 40, 25],
    "LIFETIME_DISTANCE":         [30000, 14000, 70000, 45000, 12000, 80000, 50000, 25000],
    "LIFETIME_POINTS_EARNED":    [40000, 22000, 100000, 60000, 15000, 110000, 70000, 35000],
    "LIFETIME_POINTS_REDEEMED":  [1000, 0, 800, 600, 0, 1200, 400, 200],
    "CHURN_FLAG":      [0, 1, 0, 1, 0, 1, 0, 1],
})


@pytest.fixture
def sqlite_factory(monkeypatch):
    """A conn_factory that hands back a SQLite connection with SAMPLE loaded."""
    conn = sqlite3.connect(":memory:")
    SAMPLE.to_sql("MASTER_CHURN_FEATURES", conn, index=False)
    monkeypatch.setenv("SNOWFLAKE_USER", "u")
    monkeypatch.setenv("SNOWFLAKE_PASSWORD", "p")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acc")
    return lambda: conn


def test_returns_three_tuple(sqlite_factory):
    X, y, feats = pull_snowflake.fetch_and_clean_data(conn_factory=sqlite_factory)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(feats, list)
    assert "CHURN_FLAG" not in feats
    assert "LOYALTY_NUMBER" not in feats
    assert "SALARY" in feats


def test_negative_salary_treated_as_missing(sqlite_factory):
    X, _, _ = pull_snowflake.fetch_and_clean_data(conn_factory=sqlite_factory)
    # Row 1 has -1000 (sentinel), row 2 has None.
    # Median of valid (>=0) salaries: median([80000, 60000]) = 70000.
    assert X["SALARY"].min() >= 0
    assert X["SALARY"].iloc[1] == 70000.0
    assert X["SALARY"].iloc[2] == 70000.0


def test_dummy_columns_are_int(sqlite_factory):
    X, _, _ = pull_snowflake.fetch_and_clean_data(conn_factory=sqlite_factory)
    dummy_cols = [c for c in X.columns
                  if any(c.startswith(p) for p in ("GENDER_", "EDUCATION_",
                                                  "MARITAL_STATUS_",
                                                  "LOYALTY_CARD_",
                                                  "ENROLLMENT_TYPE_"))]
    assert dummy_cols, "no dummy columns produced"
    for c in dummy_cols:
        assert X[c].dtype.kind in "iu", f"{c} is not int: {X[c].dtype}"


def test_feature_names_align_with_columns(sqlite_factory):
    X, y, feats = pull_snowflake.fetch_and_clean_data(conn_factory=sqlite_factory)
    assert list(X.columns) == feats
    assert len(feats) == 16  # 6 numeric + 10 dummies (drop_first=True)


def test_missing_env_raises(monkeypatch):
    def _boom():
        raise RuntimeError("missing env")
    monkeypatch.setattr(pull_snowflake, "_load_env", _boom)
    with pytest.raises(RuntimeError, match="missing env"):
        pull_snowflake.fetch_and_clean_data(conn_factory=lambda: None)
