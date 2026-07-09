"""
Local dry-run: rebuilds the Snowflake `MASTER_CHURN_FEATURES` table
from the raw CSVs in `data/raw/` and writes the result to a local
SQLite database. This is the path the production retrain will follow
once `SNOWFLAKE_*` env vars are set; running this script verifies
the data pipeline without any external dependency.

Run with:
    myenv/Scripts/python.exe scripts/local_dryrun.py
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("local_dryrun")

RAW_DIR = ROOT / "data" / "raw"
OUT_DB = ROOT / "data" / "local_master.db"


def _build_lifetime_features(flights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-month flight activity into the 4 lifetime features
    the production pipeline expects."""
    flights = flights.copy()
    flights.columns = [c.strip().upper().replace(" ", "_") for c in flights.columns]
    agg = flights.groupby("LOYALTY_NUMBER").agg(
        LIFETIME_FLIGHTS=("TOTAL_FLIGHTS", "sum"),
        LIFETIME_DISTANCE=("DISTANCE", "sum"),
        LIFETIME_POINTS_EARNED=("POINTS_ACCUMULATED", "sum"),
        LIFETIME_POINTS_REDEEMED=("POINTS_REDEEMED", "sum"),
    ).reset_index()
    return agg


def build_master_table() -> pd.DataFrame:
    """Reproduce what Snowflake's MASTER_CHURN_FEATURES would look like.

    Column names are normalized to UPPERCASE to match what the
    production pipeline expects (Snowflake returns uppercase by
    default; CSVs are title-case).
    """
    loyalty = pd.read_csv(RAW_DIR / "Customer Loyalty History.csv")
    flights = pd.read_csv(RAW_DIR / "Customer Flight Activity.csv")

    # Normalize column names to UPPERCASE
    loyalty.columns = [c.strip().upper().replace(" ", "_") for c in loyalty.columns]
    flights.columns = [c.strip().upper().replace(" ", "_") for c in flights.columns]

    logger.info("Raw loyalty rows: %d, flight rows: %d",
                len(loyalty), len(flights))

    lifetime = _build_lifetime_features(flights)
    master = loyalty.merge(lifetime, on="LOYALTY_NUMBER", how="left")

    # CHURN_FLAG: 1 if Cancellation Year is non-null
    master["CHURN_FLAG"] = master["CANCELLATION_YEAR"].notna().astype(int)

    # Drop the cancellation columns — they're a label leak
    master = master.drop(columns=["CANCELLATION_YEAR", "CANCELLATION_MONTH"])

    # Fill the lifetime columns with 0 for customers with no flight activity
    for c in ("LIFETIME_FLIGHTS", "LIFETIME_DISTANCE",
              "LIFETIME_POINTS_EARNED", "LIFETIME_POINTS_REDEEMED"):
        master[c] = master[c].fillna(0).astype(int)

    logger.info("Master table: %d rows, %d columns", len(master), len(master.columns))
    logger.info("Churn rate: %.2f%%", master["CHURN_FLAG"].mean() * 100)
    return master


def main() -> int:
    master = build_master_table()
    conn = sqlite3.connect(OUT_DB)
    master.to_sql("MASTER_CHURN_FEATURES", conn, if_exists="replace", index=False)
    conn.close()
    logger.info("Wrote %d rows to %s", len(master), OUT_DB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
