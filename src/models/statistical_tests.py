"""
Expanded statistical testing for the Airline Churn project.

Tests included:
  - Welch's t-test            (means, robust to unequal variance)
  - Mann-Whitney U            (non-parametric, robust to salary outliers)
  - Levene's test             (variance equality)
  - Chi-square independence  (categorical vs churn)
  - Cramér's V                (effect size for chi-square)
  - Bootstrap CI on recall    (uncertainty around the headline metric)
  - Per-segment recall        (Star / Nova / Aurora cards)

The notebook `notebooks/statistical_testing.ipynb` only ran a t-test on
LIFETIME_FLIGHTS. This module formalizes the full battery of tests a
stakeholder would ask for, and is wired into `make stats`.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import recall_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.data_pipeline.pull_snowflake import fetch_and_clean_data  # noqa: E402

logger = logging.getLogger(__name__)

NUMERIC_COLS = [
    "SALARY", "CLV", "LIFETIME_FLIGHTS", "LIFETIME_DISTANCE",
    "LIFETIME_POINTS_EARNED", "LIFETIME_POINTS_REDEEMED",
]
CATEGORICAL_COLS = [
    "GENDER", "EDUCATION", "MARITAL_STATUS", "LOYALTY_CARD", "ENROLLMENT_TYPE",
]


def _cramers_v(confusion: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(confusion, correction=False).statistic
    n = confusion.sum()
    r, k = confusion.shape
    if n == 0 or min(r, k) <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def compare_numeric(df: pd.DataFrame, col: str, y_col: str = "CHURN_FLAG") -> Dict[str, Any]:
    g0 = df.loc[df[y_col] == 0, col].dropna()
    g1 = df.loc[df[y_col] == 1, col].dropna()
    if len(g0) < 2 or len(g1) < 2:
        return {"feature": col, "error": "insufficient samples"}
    t_stat, t_p = stats.ttest_ind(g0, g1, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    lv_stat, lv_p = stats.levene(g0, g1)
    return {
        "feature": col,
        "mean_retained": float(g0.mean()),
        "mean_churned": float(g1.mean()),
        "median_retained": float(g0.median()),
        "median_churned": float(g1.median()),
        "welch_t_p": float(t_p),
        "mannwhitney_p": float(u_p),
        "levene_p": float(lv_p),
        "significant_005": bool(min(t_p, u_p) < 0.05),
    }


def compare_categorical(df: pd.DataFrame, col: str, y_col: str = "CHURN_FLAG") -> Dict[str, Any]:
    ct = pd.crosstab(df[col], df[y_col]).values
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"feature": col, "error": "insufficient categories"}
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return {
        "feature": col,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": _cramers_v(ct),
        "significant_005": bool(p < 0.05),
    }


def bootstrap_recall_ci(y_true: np.ndarray, y_pred: np.ndarray,
                        n_boot: int = 1000, alpha: float = 0.05,
                        seed: int = 42) -> Tuple[float, float, float]:
    """Returns (point_estimate, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = recall_score(y_true[idx], y_pred[idx], zero_division=0)
    point = recall_score(y_true, y_pred, zero_division=0)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def per_segment_recall(df_with_pred: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    rows = []
    for seg, g in df_with_pred.groupby(segment_col):
        if g["y_true"].nunique() < 2:
            continue
        rec = recall_score(g["y_true"], g["y_pred"], zero_division=0)
        rows.append({"segment": str(seg), "n": len(g),
                     "recall": rec,
                     "churn_rate": float(g["y_true"].mean())})
    return pd.DataFrame(rows).sort_values("recall", ascending=False)


def run_all() -> pd.DataFrame:
    """
    Pulls the *raw* (un-encoded) data. Assumes Snowflake's
    MASTER_CHURN_FEATURES still contains the original categoricals
    alongside CHURN_FLAG. If not, the categorical block silently returns
    empty rows.
    """
    X, y, _ = fetch_and_clean_data()
    df = X.assign(CHURN_FLAG=y)

    rows: List[Dict[str, Any]] = []
    for col in NUMERIC_COLS:
        if col in df.columns:
            rows.append(compare_numeric(df, col))
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            rows.append(compare_categorical(df, col))

    out = pd.DataFrame(rows)
    logger.info("Statistical test summary:\n%s", out.to_string(index=False))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_all()
