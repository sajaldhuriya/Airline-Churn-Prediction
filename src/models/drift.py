"""
Data Quality Audit for the AeroRetain AI production model.

A standalone script that catches the moment the world stops looking
like the data we trained on. If `LIFETIME_FLIGHTS` shifts because the
airline launched a new short-haul route that floods the population
with first-time flyers, the SVC's decision boundary is now
miscalibrated. The next retrain cycle will fix it, but a customer-
clobbering miscalibration in the meantime is what we want to avoid.

Run it on a schedule (cron, GitHub Actions, Airflow — your choice) and
treat the resulting `drift_report.json` as a gate. Locally you can
just run `make data-audit`.

Method:
  - Numeric features: two-sample Kolmogorov-Smirnov test. Non-parametric,
    makes no normality assumption, returns a [0, 1] statistic that
    directly quantifies *how much* the two CDFs diverge.
  - Categorical features: chi-square test on value proportions, with
    Cramér's V as the effect size (same idea as T-003 in the stats
    battery, but applied to recent-vs-baseline instead of
    churn-vs-retained).

Output:
  - `drift_report.json` with one row per feature: {feature, kind,
    statistic, p_value, severity, status}. Severity bins:
        negligible  statistic < 0.10
        small       0.10 <= statistic < 0.20
        medium      0.20 <= statistic < 0.30
        large       statistic >= 0.30
  - Status is "drift" if p_value < 0.05 AND severity != "negligible".
    Either alone is fine; both together is the actionable signal.
  - The top-level "any_drift" flag flips True if any feature is in drift.
    CI / schedulers can branch on it to fail the run and page a human;
    locally you can `cat drift_report.json` after `make data-audit`.

Inputs:
  - baseline: pd.DataFrame of training-time features (or any
    "reference" population). Must have the same columns as recent.
  - recent: pd.DataFrame of features for the period under test.
  - numeric_cols / categorical_cols: explicit column lists. We don't
    infer dtype because int-encoded dummies (e.g. GENDER_Male) are
    numeric in pandas but categorical in spirit.

This is hermetic: it takes two DataFrames and writes a JSON file. No
MLflow, no live API, no S3.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Severity bins. KS-statistic (or Cramér's V) is in [0, 1], where:
#   < 0.10  — visually identical CDFs / proportions
#   0.10-0.20 — slight separation
#   0.20-0.30 — clearly different
#   >= 0.30 — major shift
_SEVERITY_BINS = (
    (0.10, "negligible"),
    (0.20, "small"),
    (0.30, "medium"),
    (float("inf"), "large"),
)

# We treat a feature as "in drift" only if BOTH conditions hold:
#   (a) the p-value is below the alpha (5%), and
#   (b) the effect size is at least "small".
# A tiny p-value on a negligible effect is the n=16K statistical-power
# trap that the stats battery warns about (see T-001/T-003 in
# business_strategy/flask_dashboard/data_access.py).
_ALPHA = 0.05
_MIN_SEVERITY_FOR_DRIFT = "small"


@dataclass
class FeatureDrift:
    feature: str
    kind: str           # "numeric" or "categorical"
    statistic: float    # KS statistic or Cramér's V
    p_value: float
    severity: str       # negligible / small / medium / large
    status: str         # "drift" or "stable"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    baseline_n: int
    recent_n: int
    alpha: float
    features: List[FeatureDrift]

    @property
    def any_drift(self) -> bool:
        return any(f.status == "drift" for f in self.features)

    @property
    def drifted_features(self) -> List[str]:
        return [f.feature for f in self.features if f.status == "drift"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_n": self.baseline_n,
            "recent_n":   self.recent_n,
            "alpha":      self.alpha,
            "any_drift":  self.any_drift,
            "drifted_features": self.drifted_features,
            "features":   [f.to_dict() for f in self.features],
        }


def _bin_severity(stat: float) -> str:
    for threshold, name in _SEVERITY_BINS:
        if stat < threshold:
            return name
    return "large"  # unreachable given the inf bin, but defensive


def _is_drift(stat: float, p_value: float) -> bool:
    severity = _bin_severity(stat)
    severity_ok = _SEVERITY_ORDER[severity] >= _SEVERITY_ORDER[_MIN_SEVERITY_FOR_DRIFT]
    return bool(p_value < _ALPHA) and severity_ok


# Numeric encoding so we can compare severity labels.
_SEVERITY_ORDER = {name: i for i, (_, name) in enumerate(_SEVERITY_BINS)}


def _ks_numeric(baseline: pd.Series, recent: pd.Series) -> FeatureDrift:
    a = baseline.dropna().to_numpy(dtype=float)
    b = recent.dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return FeatureDrift(
            feature=baseline.name or "",
            kind="numeric",
            statistic=0.0,
            p_value=1.0,
            severity="negligible",
            status="stable",
        )
    ks_stat, p_value = stats.ks_2samp(a, b)
    stat = float(ks_stat)
    p = float(p_value)
    return FeatureDrift(
        feature=baseline.name or "",
        kind="numeric",
        statistic=stat,
        p_value=p,
        severity=_bin_severity(stat),
        status="drift" if _is_drift(stat, p) else "stable",
    )


def _chi2_categorical(baseline: pd.Series, recent: pd.Series) -> FeatureDrift:
    """Chi-square on value proportions + Cramér's V as effect size.

    Bucketing: count proportions of each unique value in baseline and
    recent, align the categories, and run chi2_contingency on the
    resulting 2×k table. Any value present in only one side gets a 0
    in the other, so a brand-new category in production is automatically
    flagged.
    """
    name = baseline.name or ""
    base_counts = baseline.value_counts(dropna=False)
    rec_counts = recent.value_counts(dropna=False)
    cats = sorted(set(base_counts.index) | set(rec_counts.index), key=str)

    # 2 rows: [baseline, recent]; one column per category.
    table = np.array([
        [int(base_counts.get(c, 0)) for c in cats],
        [int(rec_counts.get(c, 0))   for c in cats],
    ], dtype=float)

    # Degenerate cases: all-zero row, or a single category. KS-style
    # "no variability" → report stable, no drift to flag.
    if table.sum() == 0 or (table > 0).sum() < 2:
        return FeatureDrift(name, "categorical", 0.0, 1.0, "negligible", "stable")

    # If expected counts have any zeros, scipy emits a warning + falls
    # back to the asymptotic chi-square. That's fine for our purposes
    # (we only care about the magnitude of the effect).
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    # Cramér's V for a 2×k table: sqrt(chi2 / (n * (k-1))). k = cols
    # with any mass; if k == 1, V is undefined → return 0.
    k = int((table > 0).any(axis=0).sum())
    v = float(np.sqrt(chi2 / (n * max(k - 1, 1)))) if k > 1 and n > 0 else 0.0
    return FeatureDrift(
        feature=name,
        kind="categorical",
        statistic=v,
        p_value=float(p_value),
        severity=_bin_severity(v),
        status="drift" if _is_drift(v, float(p_value)) else "stable",
    )


def detect_drift(
    baseline: pd.DataFrame,
    recent: pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    alpha: float = _ALPHA,
) -> DriftReport:
    """Compare `recent` against `baseline` and return a DriftReport.

    The two frames must share every column listed in numeric_cols and
    categorical_cols. Extra columns are ignored (so passing the full
    feature DataFrame is fine).
    """
    report = DriftReport(
        baseline_n=int(len(baseline)),
        recent_n=int(len(recent)),
        alpha=alpha,
        features=[],
    )
    for col in numeric_cols:
        if col not in baseline.columns or col not in recent.columns:
            logger.warning("numeric column %r missing from one of the frames; skipped", col)
            continue
        report.features.append(_ks_numeric(baseline[col], recent[col]))
    for col in categorical_cols:
        if col not in baseline.columns or col not in recent.columns:
            logger.warning("categorical column %r missing from one of the frames; skipped", col)
            continue
        report.features.append(_chi2_categorical(baseline[col], recent[col]))

    if report.any_drift:
        logger.warning(
            "Drift detected on %d feature(s): %s",
            len(report.drifted_features),
            ", ".join(report.drifted_features),
        )
    return report


def write_report(report: DriftReport, path: str) -> str:
    """Write the report to JSON. Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# CLI: python -m src.models.drift
# ---------------------------------------------------------------------------
def main(
    recent_path: str,
    out_path: str = "drift_report.json",
    *,
    baseline_frac: float = 0.5,
) -> DriftReport:
    """Run drift on a recent CSV against a baseline pulled from the
    training pipeline. The first half of the training data is treated
    as baseline; the second half as recent. Real production use would
    pass the actual recent batch and a saved baseline instead.
    """
    from src.data_pipeline.pull_snowflake import (  # noqa: PLC0415
        fetch_and_clean_data,
    )
    logger.info("Loading training data (this can take a while)...")
    X, _, _ = fetch_and_clean_data()
    n = len(X)
    cut = int(n * baseline_frac)
    baseline = X.iloc[:cut]
    if recent_path and os.path.exists(recent_path):
        logger.info("Loading recent batch from %s", recent_path)
        recent_df = pd.read_csv(recent_path)
    else:
        logger.info("No recent CSV provided; using the second half of the training data as 'recent'")
        recent_df = X.iloc[cut:]

    # The same column lists the model was trained on. Numeric = everything
    # that isn't a dummy. Categorical = the dummies.
    numeric_cols = [
        "SALARY", "CLV", "LIFETIME_FLIGHTS", "LIFETIME_DISTANCE",
        "LIFETIME_POINTS_EARNED", "LIFETIME_POINTS_REDEEMED",
    ]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    report = detect_drift(baseline, recent_df, numeric_cols, categorical_cols)
    write_report(report, out_path)
    if report.any_drift:
        logger.error("Drift detected. See %s for the report.", out_path)
    else:
        logger.info("No drift detected. Report at %s", out_path)
    return report


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recent-csv", default="",
                   help="Path to a CSV of recent features. Empty = use the second half of training data.")
    p.add_argument("--out", default="drift_report.json",
                   help="Where to write the report (default: drift_report.json).")
    p.add_argument("--baseline-frac", type=float, default=0.5,
                   help="Fraction of training data to use as baseline (default: 0.5).")
    args = p.parse_args()
    main(args.recent_csv, args.out, baseline_frac=args.baseline_frac)
