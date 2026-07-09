"""
Robust Snowflake extraction with deterministic one-hot encoding.

Returns (X, y, feature_names). The third element is the EXACT ordered list
of column names the scaler and model were trained on. The inference layer
in `src.api.app` reads the same list to prevent train/serve skew.

Differences vs the original v1.0:
  - Negative salaries (data-quality issue: min observed is -$58,486) are
    treated as missing and imputed with the median.
  - One-hot dummies are cast to int (not bool) so the JSON schema in
    the FastAPI Pydantic model is unambiguous.
  - The Snowflake warehouse / database / schema are env-configurable.
  - The connector is wrapped in a factory `conn_factory` for hermetic tests.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Single source of truth for the categorical columns we one-hot encode.
CATEGORICAL_COLS: List[str] = [
    "GENDER",
    "EDUCATION",
    "MARITAL_STATUS",
    "LOYALTY_CARD",
    "ENROLLMENT_TYPE",
]

# Identifier / leak columns we drop before training.
DROP_COLS: List[str] = ["LOYALTY_NUMBER"]


def _load_env() -> None:
    """Load .env from the project root, regardless of CWD."""
    env_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    )
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            f".env not found at {env_path}. Copy .env.example to .env and fill in."
        )
    load_dotenv(env_path, override=False)

    required = ["SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_ACCOUNT"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {missing}")


def _default_connect():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "AIRLINE_LOYALTY_DB"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "RAW_DATA"),
        login_timeout=30,
        network_timeout=30,
    )


def fetch_and_clean_data(
    conn_factory: Optional[Callable] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Pulls the master table, cleans it, and returns:
        X              : feature DataFrame
        y              : label Series (int 0/1)
        feature_names  : exact ordered list of X.columns

    `conn_factory` is a dependency-injection seam for tests.
    """
    _load_env()
    logger.info("Downloading from Snowflake...")

    factory = conn_factory or _default_connect
    conn = factory()
    try:
        df = pd.read_sql("SELECT * FROM MASTER_CHURN_FEATURES;", conn)
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    if "CHURN_FLAG" not in df.columns:
        raise RuntimeError(
            "CHURN_FLAG missing from Snowflake result. "
            f"Got: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # 1. Salary cleanup: negative values are sentinel data-quality errors.
    #    Treat them as missing and impute with the median of the *valid*
    #    salaries, not the full column.
    # ------------------------------------------------------------------
    if "SALARY" in df.columns:
        valid_salary_mask = df["SALARY"] >= 0
        if (~valid_salary_mask).any():
            logger.warning(
                "Found %d negative-salary rows; treating as missing.",
                int((~valid_salary_mask).sum()),
            )
            df.loc[~valid_salary_mask, "SALARY"] = pd.NA
        if df["SALARY"].isna().any():
            median = df["SALARY"].median()
            df["SALARY"] = df["SALARY"].fillna(median)
            logger.info("Imputed SALARY with median %.2f", median)

    # ------------------------------------------------------------------
    # 2. Drop identifier / leak columns.
    # ------------------------------------------------------------------
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ------------------------------------------------------------------
    # 3. Deterministic one-hot encoding. drop_first=True avoids the
    #    dummy-variable trap and yields a stable column set.
    # ------------------------------------------------------------------
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Cast all dummy columns to int (pandas 2.x emits bool by default)
    dummy_cols = [
        c for c in df_encoded.columns
        if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_COLS)
    ]
    df_encoded[dummy_cols] = df_encoded[dummy_cols].astype(int)

    feature_names = [c for c in df_encoded.columns if c != "CHURN_FLAG"]
    X = df_encoded[feature_names].copy()
    y = df_encoded["CHURN_FLAG"].astype(int)

    logger.info(
        "Pipeline success. X shape=%s, churn rate=%.3f, features=%d",
        X.shape, y.mean(), len(feature_names),
    )
    return X, y, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    X, y, feats = fetch_and_clean_data()
    print(f"✅ Pipeline Success! X shape: {X.shape}, features: {len(feats)}")
