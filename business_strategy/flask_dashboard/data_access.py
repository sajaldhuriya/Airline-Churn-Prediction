"""
Live data access for the manager dashboard.

This is the seam between the Flask UI and the operational data store.
We separate this from `app.py` so the dashboard is testable without a
live Snowflake connection.

The data path is:

  data/raw/Customer Loyalty History.csv    (demographics + churn label)
            |
            |  join on Loyalty Number
            v
  data/raw/Customer Flight Activity.csv   (lifetime flights / distance / points)

For production we still want Snowflake — set DATA_BACKEND=snowflake
and wire the queries. For now the local path returns real numbers
computed from the source CSVs, so the dashboard tells the truth
even offline. If either CSV is missing, we fall back to a hard-coded
snapshot so the page still renders.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# business_strategy/flask_dashboard/data_access.py  ->  project root is 2 levels up
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOCAL_LOYALTY_CSV = os.path.join(ROOT, "data", "raw", "Customer Loyalty History.csv")
LOCAL_FLIGHT_CSV  = os.path.join(ROOT, "data", "raw", "Customer Flight Activity.csv")

# ---------------------------------------------------------------------------
# Hard-coded snapshot used only when the local CSVs can't be read.
# ---------------------------------------------------------------------------
_SNAPSHOT_KPIS = {
    "total_passengers": 14250,
    "churn_rate":       0.164,
    "churn_rate_ci":    {"point": 0.164, "lo": 0.158, "hi": 0.170, "half_width": 0.006},
    "revenue_at_risk":  "$2.4M",
    "avg_flights":      28.4,
    "card_churn": {
        "labels": ["Aurora", "Nova", "Star"],
        "values": [12.3, 18.1, 21.4],
    },
    "drivers": {
        "labels": ["Low engagement", "Points hoarded", "Service complaints", "Competitor offer"],
        "values": [42, 28, 18, 12],
    },
}

_SNAPSHOT_BCG = {"stars": 12, "cash_cows": 45, "question_marks": 18, "dogs": 25}
_SNAPSHOT_TESTS: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_local_frame() -> pd.DataFrame:
    """Build the working DataFrame from local CSVs.

    Returns columns: LOYALTY_NUMBER, GENDER, EDUCATION, MARITAL_STATUS,
    LOYALTY_CARD, SALARY, CLV, ENROLLMENT_TYPE, CHURN_FLAG, LIFETIME_FLIGHTS,
    LIFETIME_DISTANCE, LIFETIME_POINTS_EARNED, LIFETIME_POINTS_REDEEMED.
    """
    loyalty = pd.read_csv(LOCAL_LOYALTY_CSV)
    loyalty["CHURN_FLAG"] = loyalty["Cancellation Year"].notna().astype(int)
    loyalty = loyalty.rename(columns={
        "Loyalty Number":  "LOYALTY_NUMBER",
        "Gender":          "GENDER",
        "Education":       "EDUCATION",
        "Salary":          "SALARY",
        "Marital Status":  "MARITAL_STATUS",
        "Loyalty Card":    "LOYALTY_CARD",
        "Enrollment Type": "ENROLLMENT_TYPE",
    })

    flights = pd.read_csv(LOCAL_FLIGHT_CSV)
    lifetime = (flights.groupby("Loyalty Number", as_index=False)
                .agg(LIFETIME_FLIGHTS=("Total Flights", "sum"),
                     LIFETIME_DISTANCE=("Distance", "sum"),
                     LIFETIME_POINTS_EARNED=("Points Accumulated", "sum"),
                     LIFETIME_POINTS_REDEEMED=("Points Redeemed", "sum"))
                .rename(columns={"Loyalty Number": "LOYALTY_NUMBER"}))

    df = loyalty.merge(lifetime, on="LOYALTY_NUMBER", how="left")
    # Fill lifetime columns with 0 for customers with no recorded flights
    for col in ("LIFETIME_FLIGHTS", "LIFETIME_DISTANCE",
                "LIFETIME_POINTS_EARNED", "LIFETIME_POINTS_REDEEMED"):
        df[col] = df[col].fillna(0)
    return df


def _has_local_data() -> bool:
    return os.path.exists(LOCAL_LOYALTY_CSV) and os.path.exists(LOCAL_FLIGHT_CSV)


def _filter(df: pd.DataFrame, *, gender: str = "all",
            card: str = "all", education: str = "all") -> pd.DataFrame:
    """Apply the slicer filters used by both / and /api/overview."""
    out = df
    if gender and gender != "all":
        out = out[out["GENDER"].str.lower() == gender.lower()]
    if card and card != "all":
        out = out[out["LOYALTY_CARD"] == card]
    if education and education != "all":
        out = out[out["EDUCATION"] == education]
    return out


# ---------------------------------------------------------------------------
# Driver extraction from the trained champion model
# ---------------------------------------------------------------------------
# Map raw column names -> human-readable labels for the dashboard.
# Kept here (not in templates) so a single source of truth governs what
# "LIFETIME_POINTS_REDEEMED" means to a stakeholder. Mirrors the 19
# columns saved to `models/feature_names.txt` (4 engagement numerics +
# 15 dummies). SALARY and CLV were dropped in v2.0.0 (Cohen's d ≈ 0);
# every dummy for every categorical level the model was trained on is
# present here.
_DRIVER_LABELS = {
    "LIFETIME_FLIGHTS":                "Low engagement",
    "LIFETIME_DISTANCE":               "Low travel activity",
    "LIFETIME_POINTS_EARNED":          "Points hoarded (not earned)",
    "LIFETIME_POINTS_REDEEMED":        "Few redemptions",
    "GENDER_Female":                   "Gender: Female",
    "GENDER_Male":                     "Gender: Male",
    "EDUCATION_Bachelor":              "Education: Bachelor",
    "EDUCATION_College":               "Education: College",
    "EDUCATION_Doctor":                "Education: Doctorate",
    "EDUCATION_High School or Below":  "Education: High School or below",
    "EDUCATION_Master":                "Education: Master's",
    "MARITAL_STATUS_Divorced":         "Marital: Divorced",
    "MARITAL_STATUS_Married":          "Marital: Married",
    "MARITAL_STATUS_Single":           "Marital: Single",
    "LOYALTY_CARD_Aurora":             "Card: Aurora tier",
    "LOYALTY_CARD_Nova":               "Card: Nova tier",
    "LOYALTY_CARD_Star":               "Card: Star tier",
    "ENROLLMENT_TYPE_2018 Promotion":  "2018 Promotion cohort",
    "ENROLLMENT_TYPE_Standard":        "Standard enrollment",
}

# Top-N features shown in the dashboard donut/bar chart. Four keeps it
# readable; the rest are summarised as "Other" in the template.
_DRIVER_TOP_N = 4

# Bootstrap CI on the churn rate. 1000 resamples is the convention for
# 95% percentile CIs on a Bernoulli proportion; takes ~50ms on n=16k.
_BOOTSTRAP_RESAMPLES = 1000
_BOOTSTRAP_ALPHA = 0.05  # 95% CI
_BOOTSTRAP_SEED = 42  # reproducibility for the stakeholder demo


def _resolve_mlflow_uri() -> str:
    db_path = os.path.abspath(os.path.join(ROOT, "mlflow.db"))
    return f"sqlite:///{db_path}"


def _load_champion_coefficients() -> Optional[Tuple[List[str], np.ndarray]]:
    """Load champion feature importances from MLflow.

    The v2.0.0 champion is SVC(rbf, balanced, C=1.0). rbf kernels do
    NOT expose coef_ — they live in an infinite-dimensional space — so
    the only way to surface "what matters" from the trained model is
    permutation importance (computed in `train_svc.py` and logged as
    `permutation_importances.json`).

    Returns (feature_names, importance_means) or None if the artifact
    is missing or the experiment is empty. The dashboard then falls
    back to the snapshot drivers dict.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed; using snapshot drivers")
        return None

    try:
        mlflow.set_tracking_uri(_resolve_mlflow_uri())
        exp = mlflow.get_experiment_by_name("Airline_Churn_Production")
        if exp is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED' and metrics.recall > 0",
            order_by=["metrics.recall DESC"],
        )
        if runs.empty:
            return None
        run_id = runs.iloc[0]["run_id"]

        # Read the permutation-importance artifact the training script
        # logs. If it's missing (legacy run), fall through to None so
        # the snapshot is used.
        try:
            art = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="permutation_importances.json",
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "permutation_importances.json missing on champion run; "
                "using snapshot drivers"
            )
            return None

        with open(art, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            return None

        feats = [r["feature"] for r in records]
        mags = np.array([float(r.get("importance_mean", 0.0)) for r in records])
        if not feats or mags.size == 0:
            return None

        return feats, mags
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load champion drivers (%s); using snapshot", exc)
        return None


@lru_cache(maxsize=1)
def get_model_drivers() -> Dict[str, Any]:
    """Top-N churn drivers from the trained champion, as a chart-ready dict.

    Returns {labels: [...], values: [...]} where values are *percentages*
    of total |coef|, summing to 100. Falls back to the snapshot dict if
    the champion model cannot be loaded (e.g. dev env without MLflow).
    Cached because the model doesn't change between requests.
    """
    snapshot = _SNAPSHOT_KPIS["drivers"]
    loaded = _load_champion_coefficients()
    if loaded is None:
        return snapshot

    feats, mags = loaded
    # Permutation importances are signed: negative means "shuffling
    # this feature HELPS the model", which is noise, not a real
    # driver of churn. Clip to non-negative before ranking so the
    # dashboard only surfaces features whose absence hurts the model.
    pos_mags = np.maximum(mags, 0.0)
    # Rank descending by magnitude, take top N
    order = np.argsort(pos_mags)[::-1]
    top_idx = order[:_DRIVER_TOP_N]
    top_mags = pos_mags[top_idx]
    total = top_mags.sum()
    if total <= 0:
        return snapshot

    # Normalize to integer percentages so the donut chart's labels are
    # readable. We round-then-fix to make sure they sum to exactly 100.
    raw_pct = (top_mags / total * 100).round().astype(int)
    diff = 100 - int(raw_pct.sum())
    if diff != 0:
        # Apply the rounding remainder to the largest bucket so the
        # visual is faithful to the magnitude ordering.
        raw_pct[0] = int(raw_pct[0]) + diff

    labels = [_DRIVER_LABELS.get(feats[i], feats[i]) for i in top_idx]
    return {"labels": labels, "values": [int(v) for v in raw_pct]}


# ---------------------------------------------------------------------------
# Per-segment recall — mirrors notebooks/machine_learning.ipynb §7
# ---------------------------------------------------------------------------
# Aggregate recall can hide very different behaviour across sub-populations.
# A model that scores 0.84 overall can still be 0.95 on Star-card holders
# and 0.50 on 2018-Promotion customers — exactly the information the
# strategy team needs to know which segments to retrain on first.
#
# Approach:
#   1. Load the champion from MLflow (cached on first call).
#   2. Build a 19-column feature row for every customer in the local CSV.
#      The categoricals are one-hot encoded with the exact column set
#      from `models/feature_names.txt`; the numeric engagement columns
#      pass through.
#   3. Run `model.predict()` on the full frame.
#   4. Group by the requested segment column and compute recall per
#      segment. Skip segments with fewer than `min_n` rows or only one
#      churn-status value (recall is undefined there).
#
# Falls back to an empty list when the champion can't be loaded. The
# dashboard then surfaces the empty state instead of crashing.
_SEGMENT_MIN_N = 30
_SEGMENT_COLS = ("LOYALTY_CARD", "EDUCATION", "GENDER",
                 "MARITAL_STATUS", "ENROLLMENT_TYPE")

# Categorical levels the model was trained on, in display order. Anything
# in the source CSV outside this set is dropped from the segment table
# (we can't model a category the model has never seen).
_SEGMENT_LEVELS = {
    "LOYALTY_CARD":     ["Star", "Nova", "Aurora"],
    "EDUCATION":        ["Bachelor", "College", "Doctor", "High School or Below", "Master"],
    "GENDER":           ["Female", "Male"],
    "MARITAL_STATUS":   ["Divorced", "Married", "Single"],
    "ENROLLMENT_TYPE":  ["2018 Promotion", "Standard"],
}


@lru_cache(maxsize=1)
def _load_champion_for_segments() -> Optional[Tuple[Any, List[str]]]:
    """Load the champion Pipeline and its feature schema from MLflow.

    Returns (pipeline, feature_names) or None. Cached because the
    model doesn't change between requests and loading + serialization
    is the expensive part of per-segment recall.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("mlflow not installed; per-segment recall disabled")
        return None

    try:
        mlflow.set_tracking_uri(_resolve_mlflow_uri())
        exp = mlflow.get_experiment_by_name("Airline_Churn_Production")
        if exp is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED' and metrics.recall > 0",
            order_by=["metrics.recall DESC"],
        )
        if runs.empty:
            return None
        run_id = runs.iloc[0]["run_id"]
        pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/tuned_model")

        # Read the sidecar feature-name list — it tells us the *exact*
        # dummified column set the model was trained on.
        try:
            art = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="feature_names.txt",
            )
            with open(art, "r", encoding="utf-8") as f:
                feature_names = [line.strip() for line in f if line.strip()]
        except Exception:  # noqa: BLE001
            feature_names = list(getattr(
                pipeline.named_steps.get("scaler"), "feature_names_in_", []
            ))

        if not feature_names:
            return None
        return pipeline, feature_names
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load champion for segments (%s); disabled", exc)
        return None


def _build_segment_feature_frame(
    df: pd.DataFrame, feature_names: List[str]
) -> pd.DataFrame:
    """Build a feature frame matching `feature_names` exactly.

    Engagement numerics pass through. Each categorical is one-hot
    encoded with the same column set the model saw at training time,
    so the order of dummies matches what the Pipeline expects.
    """
    out = pd.DataFrame(index=df.index)

    # Engagement numerics — model is trained on these four
    for col in ("LIFETIME_FLIGHTS", "LIFETIME_DISTANCE",
                "LIFETIME_POINTS_EARNED", "LIFETIME_POINTS_REDEEMED"):
        if col in feature_names and col in df.columns:
            out[col] = df[col].astype(float)

    # Categoricals — one-hot with the same prefix/sep the model used
    for cat, levels in _SEGMENT_LEVELS.items():
        if cat not in df.columns:
            continue
        for level in levels:
            col = f"{cat}_{level}"
            if col in feature_names:
                out[col] = (df[cat] == level).astype(int)

    # Reindex to the model's exact column order; missing columns become 0
    return out.reindex(columns=feature_names, fill_value=0)


def get_per_segment_recall(
    segment_col: str = "LOYALTY_CARD",
) -> List[Dict[str, Any]]:
    """Per-segment recall from the trained champion, on the local CSV.

    Returns a list of {segment, n, churn_rate, recall} dicts, sorted by
    recall ascending so the weakest segments surface first. Returns []
    if the champion can't be loaded or the source data is missing.
    Cached per segment column because the model doesn't change between
    requests and inference on the full population is the expensive bit.
    """
    return _get_per_segment_recall_uncached(segment_col)


@lru_cache(maxsize=8)
def _get_per_segment_recall_uncached(
    segment_col: str = "LOYALTY_CARD",
) -> List[Dict[str, Any]]:
    if not _has_local_data():
        return []
    if segment_col not in _SEGMENT_COLS:
        logger.warning("unknown segment column: %s", segment_col)
        return []

    loaded = _load_champion_for_segments()
    if loaded is None:
        return []
    pipeline, feature_names = loaded

    try:
        df = _load_local_frame()
    except Exception:  # noqa: BLE001
        return []

    if df.empty or df["CHURN_FLAG"].nunique() < 2:
        return []

    try:
        X = _build_segment_feature_frame(df, feature_names)
        # predict() is the SVC's calibrated hard label. With balanced
        # class weights + Platt scaling, this matches the API's
        # /predict output (label derived from a 0.5 probability
        # threshold). Recall is then the standard "fraction of true
        # churners we caught" — exactly what the notebook reports.
        y_pred = pipeline.predict(X).astype(int)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Per-segment recall inference failed: %s", exc)
        return []

    y_true = df["CHURN_FLAG"].astype(int).to_numpy()
    df_eval = df.assign(_y_true=y_true, _y_pred=y_pred)

    rows: List[Dict[str, Any]] = []
    levels = _SEGMENT_LEVELS.get(segment_col, [])
    # Iterate levels in the order the model knows them, not the
    # alphabetical order pandas would use by default. This is what
    # the stakeholder expects to see in the dashboard.
    for level in levels:
        g = df_eval[df_eval[segment_col] == level]
        if len(g) < _SEGMENT_MIN_N or g["_y_true"].nunique() < 2:
            continue
        tp = int(((g["_y_true"] == 1) & (g["_y_pred"] == 1)).sum())
        fn = int(((g["_y_true"] == 1) & (g["_y_pred"] == 0)).sum())
        pos = tp + fn
        recall = float(tp / pos) if pos > 0 else 0.0
        rows.append({
            "segment":       str(level),
            "n":             int(len(g)),
            "churn_rate":    float(g["_y_true"].mean()),
            "recall":        recall,
            "true_positives": tp,
            "false_negatives": fn,
        })

    # Weakest first — the manager wants to see the bottom of the
    # table, not the top.
    rows.sort(key=lambda r: r["recall"])
    return rows


# ---------------------------------------------------------------------------
# Bootstrap confidence interval on the churn rate
# ---------------------------------------------------------------------------
def bootstrap_churn_rate_ci(
    flags: pd.Series,
    n_resamples: int = _BOOTSTRAP_RESAMPLES,
    alpha: float = _BOOTSTRAP_ALPHA,
    seed: int = _BOOTSTRAP_SEED,
) -> Optional[Dict[str, float]]:
    """Percentile bootstrap 95% CI on the mean of a 0/1 churn series.

    Returns {point, lo, hi, half_width} or None if there are too few
    observations to resample. The point estimate is the *observed* mean,
    not the bootstrap mean — analysts expect the headline number to
    match what's printed on the tile.
    """
    arr = np.asarray(flags, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return None

    rng = np.random.default_rng(seed)
    # Resample with replacement n times, compute the mean for each
    # resample, then take the alpha/2 and 1-alpha/2 percentiles.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "point":      float(arr.mean()),
        "lo":         float(lo),
        "hi":         float(hi),
        "half_width": float((hi - lo) / 2),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_kpis(gender: str = "all", card: str = "all",
             education: str = "all") -> Dict[str, Any]:
    """Headline KPIs. `revenue_at_risk` is an estimate:
    churners * mean CLV (rounded to nearest $0.1M)."""
    if not _has_local_data():
        return _SNAPSHOT_KPIS

    try:
        df = _filter(_load_local_frame(), gender=gender, card=card, education=education)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_kpis local-load failed (%s); using snapshot", exc)
        return _SNAPSHOT_KPIS

    if df.empty:
        return {
            "total_passengers": 0,
            "churn_rate": 0.0,
            "churn_rate_ci": {"point": 0.0, "lo": 0.0, "hi": 0.0, "half_width": 0.0},
            "revenue_at_risk": "$0",
            "avg_flights": 0.0,
            "card_churn": {"labels": [], "values": []},
            "drivers": get_model_drivers(),
        }

    churn_rate = float(df["CHURN_FLAG"].mean())
    churn_ci = bootstrap_churn_rate_ci(df["CHURN_FLAG"])
    revenue_at_risk = float((df["CHURN_FLAG"] * df["CLV"]).sum())
    avg_flights = float(df["LIFETIME_FLIGHTS"].mean())

    # Churn rate by card tier (only cards that are present in the slice)
    card_group = (df.groupby("LOYALTY_CARD")["CHURN_FLAG"].mean() * 100).round(2)
    card_group = card_group.sort_index()

    return {
        "total_passengers": int(len(df)),
        "churn_rate":       churn_rate,
        "churn_rate_ci":    churn_ci,
        "revenue_at_risk":  _fmt_money(revenue_at_risk),
        "avg_flights":      avg_flights,
        "card_churn": {
            "labels": [str(c) for c in card_group.index.tolist()],
            "values": [float(v) for v in card_group.values],
        },
        "drivers": get_model_drivers(),   # from the trained champion, not a static dict
    }


def get_bcg_segments(gender: str = "all", card: str = "all",
                     education: str = "all") -> Dict[str, int]:
    """BCG segments: classify customers by CLV (median split) and flight count (median split).
    Stars = high CLV + high flights. Dogs = low CLV + low flights.
    Question Marks = low CLV + high flights. Cash Cows = high CLV + low flights.
    """
    if not _has_local_data():
        return _SNAPSHOT_BCG

    try:
        df = _filter(_load_local_frame(), gender=gender, card=card, education=education)
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_bcg_segments local-load failed (%s); using snapshot", exc)
        return _SNAPSHOT_BCG

    if df.empty or df["CLV"].nunique() < 2 or df["LIFETIME_FLIGHTS"].nunique() < 2:
        return {"stars": 0, "cash_cows": 0, "question_marks": 0, "dogs": 0}

    clv_median  = df["CLV"].median()
    flt_median  = df["LIFETIME_FLIGHTS"].median()
    high_clv    = df["CLV"] >= clv_median
    high_flight = df["LIFETIME_FLIGHTS"] >= flt_median

    n = len(df)
    return {
        "stars":         int(round((high_clv & high_flight).sum()  / n * 100)),
        "cash_cows":     int(round((high_clv & ~high_flight).sum() / n * 100)),
        "question_marks":int(round((~high_clv & high_flight).sum() / n * 100)),
        "dogs":          int(round((~high_clv & ~high_flight).sum() / n * 100)),
    }


def get_hypothesis_tests() -> List[Dict[str, Any]]:
    """Returns the latest statistical-test log for the stats page.

    The full battery of tests lives in `src.models.statistical_tests`.
    Here we hard-code a few of the most business-relevant tests with
    their observed outcomes; the live values would be replaced by an
    MLflow query once the experiment logger is in place.
    """
    if not _has_local_data():
        return _SNAPSHOT_TESTS

    try:
        df = _load_local_frame()
    except Exception:  # noqa: BLE001
        return _SNAPSHOT_TESTS

    from scipy import stats
    tests: List[Dict[str, Any]] = []

    # 1. Welch t-test: lifetime flights differ between churners / non-churners
    g0 = df.loc[df["CHURN_FLAG"] == 0, "LIFETIME_FLIGHTS"]
    g1 = df.loc[df["CHURN_FLAG"] == 1, "LIFETIME_FLIGHTS"]
    if len(g0) > 1 and len(g1) > 1:
        t_stat, p = stats.ttest_ind(g0, g1, equal_var=False)
        # Levene's test for variance equality — tells us if the two
        # groups have similar spread, which is a prerequisite assumption
        # of the equal-variance t-test (Student's). Welch's t-test does
        # NOT assume equal variance, so a Levene rejection just means
        # "use Welch, not Student." The effect size for Levene is the
        # ratio of the larger group's variance to the smaller's, which
        # is more interpretable than the W statistic on a non-familiar
        # audience.
        lev_stat, lev_p = stats.levene(g0, g1, center="median")
        v0, v1 = float(g0.var(ddof=1)), float(g1.var(ddof=1))
        var_ratio = max(v0, v1) / min(v0, v1) if min(v0, v1) > 0 else float("inf")
        if lev_p < 0.05:
            variance_note = (f"variances differ (Levene p={lev_p:.2e}, "
                             f"ratio={var_ratio:.2f}×) — Welch's t-test is the correct choice")
        else:
            variance_note = (f"variances are comparable (Levene p={lev_p:.2e}, "
                             f"ratio={var_ratio:.2f}×) — equal-variance assumption is OK")
        tests.append({
            "id": "T-001",
            "metric": "LIFETIME_FLIGHTS (churned vs retained)",
            "status": "REJECT H0" if p < 0.05 else "FAIL TO REJECT",
            "test_type": "Welch's two-sample t-test",
            "p_value": f"{p:.2e}",
            "null_hypothesis": "Mean lifetime flights are equal for retained and churned customers.",
            "insight": ("Churned customers average significantly fewer flights. "
                        "Onboarding must drive the second-flight moment."),
            "variance_check": variance_note,
        })

    # 2. Mann-Whitney U: points redeemed (non-parametric, robust to outliers)
    g0 = df.loc[df["CHURN_FLAG"] == 0, "LIFETIME_POINTS_REDEEMED"]
    g1 = df.loc[df["CHURN_FLAG"] == 1, "LIFETIME_POINTS_REDEEMED"]
    if len(g0) > 1 and len(g1) > 1:
        u_stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        tests.append({
            "id": "T-002",
            "metric": "LIFETIME_POINTS_REDEEMED (churned vs retained)",
            "status": "REJECT H0" if p < 0.05 else "FAIL TO REJECT",
            "test_type": "Mann-Whitney U (non-parametric)",
            "p_value": f"{p:.2e}",
            "null_hypothesis": "The distribution of points-redeemed is the same for both groups.",
            "insight": ("Churners redeem far fewer points. Marketing should push redemption "
                        "campaigns, not just earning campaigns."),
        })

    # 3. Chi-square: loyalty card vs churn
    ct = pd.crosstab(df["LOYALTY_CARD"], df["CHURN_FLAG"]).values
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.sum()
        # Cramér's V as effect size — tells us *how strong* the link is,
        # not just whether it exists. Ranges [0, 1]: 0.10=small, 0.30=med, 0.50=large
        cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if n > 0 else 0.0
        rejected = p < 0.05
        if rejected:
            if cramers_v >= 0.30:
                effect = f"strong (Cramér's V = {cramers_v:.2f})"
            elif cramers_v >= 0.10:
                effect = f"moderate (Cramér's V = {cramers_v:.2f})"
            else:
                effect = f"statistically significant but small in magnitude (Cramér's V = {cramers_v:.2f})"
            insight = (f"Card tier and churn ARE linked — effect is {effect}. "
                       "Segment retention plays by tier, not by the population as a whole.")
        else:
            insight = (f"Churn rate does NOT meaningfully differ by card tier (p={p:.2f}, "
                       f"Cramér's V={cramers_v:.2f}). Retention strategy can treat tiers "
                       "similarly for now, but re-check after the next retraining cycle.")
        tests.append({
            "id": "T-003",
            "metric": "LOYALTY_CARD vs CHURN_FLAG",
            "status": "REJECT H0" if rejected else "FAIL TO REJECT",
            "test_type": "Chi-square test of independence",
            "p_value": f"{p:.2e}",
            "null_hypothesis": "Churn rate is independent of loyalty card tier.",
            "insight": insight,
        })

    # 4. Levene's test (standalone): variance of lifetime flights
    #    between churned and retained. With n=16K the F/W statistic is
    #    almost always significant — what matters is the *effect size*,
    #    the variance ratio, and whether the inequality is large enough
    #    to matter for downstream modeling.
    g0 = df.loc[df["CHURN_FLAG"] == 0, "LIFETIME_FLIGHTS"]
    g1 = df.loc[df["CHURN_FLAG"] == 1, "LIFETIME_FLIGHTS"]
    if len(g0) > 1 and len(g1) > 1:
        lev_stat, lev_p = stats.levene(g0, g1, center="median")
        v0, v1 = float(g0.var(ddof=1)), float(g1.var(ddof=1))
        var_ratio = max(v0, v1) / min(v0, v1) if min(v0, v1) > 0 else float("inf")
        # Cohen's heuristic for variance ratio: ~1.5 = "somewhat different",
        # ~2 = "substantially different". Above 4 we'd flag it.
        if var_ratio >= 4.0:
            magnitude = "substantially different"
        elif var_ratio >= 1.5:
            magnitude = "moderately different"
        else:
            magnitude = "comparable"
        if lev_p < 0.05:
            var_insight = (f"Spread of lifetime flights is {magnitude} between "
                           f"churners and retainers (variance ratio = {var_ratio:.2f}×, "
                           f"Levene p={lev_p:.2e}). This is why T-001 uses Welch's "
                           "t-test rather than Student's.")
        else:
            var_insight = (f"Spread of lifetime flights is {magnitude} between groups "
                           f"(variance ratio = {var_ratio:.2f}×, Levene p={lev_p:.2e}). "
                           "Equal-variance assumption is reasonable for T-001.")
        tests.append({
            "id": "T-004",
            "metric": "LIFETIME_FLIGHTS spread (churned vs retained)",
            "status": "REJECT H0" if lev_p < 0.05 else "FAIL TO REJECT",
            "test_type": "Levene's test (median-centered)",
            "p_value": f"{lev_p:.2e}",
            "null_hypothesis": "Variance of lifetime flights is equal for both groups.",
            "insight": var_insight,
        })

    return tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_money(amount: float) -> str:
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    return f"${amount:.0f}"


def _reset_cache_for_tests() -> None:
    """Test hook: clear the cached DataFrame so tests can mutate the CSV."""
    _load_local_frame.cache_clear()
