"""
Shared fixtures for the dashboard test suite.

`data/raw/*.csv` is gitignored, so a clean CI checkout has no CSVs.
The dashboard data layer (`data_access._load_local_frame`) reads them
at test time. Without these fixtures, every dashboard test would
fall back to the snapshot dict and the assertions on live values
would silently pass for the wrong reason.

The `synthetic_csvs` fixture:
  - If the real CSVs are present (local dev), no-op.
  - Otherwise, synthesizes a small fake dataset, writes it to tmp,
    and monkeypatches the data-access module's CSV paths to point
    at it. The data layer reads through those module-level names, so
    the patch is sufficient — no need to touch the production code.
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


@pytest.fixture
def synthetic_csvs(tmp_path, monkeypatch):
    """Materialize synthetic CSVs and point the data layer at them.

    Used by every test that depends on the live data layer. If the
    real local CSVs are present, the fixture no-ops (local dev case).
    """
    if data_access._has_local_data():
        yield
        return

    rng = np.random.default_rng(0)
    n = 200
    loyalty = pd.DataFrame({
        "Loyalty Number":  np.arange(1, n + 1),
        "Gender":          rng.choice(["Male", "Female"], size=n),
        "Education":       rng.choice(["Bachelor", "Master", "College", "Doctor",
                                       "High School or Below"], size=n),
        "Salary":          np.where(rng.random(n) < 0.05, -1.0,
                                    rng.normal(70_000, 20_000, n).round(2)),
        "Marital Status":  rng.choice(["Married", "Single", "Divorced"], size=n),
        "Loyalty Card":    rng.choice(["Star", "Nova", "Aurora"], size=n),
        "CLV":             rng.normal(5000, 1500, n).round(2),
        "Enrollment Type": rng.choice(["Standard", "2018 Promotion"], size=n),
        # 15% churners so the hypothesis tests have signal to detect
        "Cancellation Year": rng.choice([np.nan, 2023.0, 2024.0], n, p=[0.85, 0.10, 0.05]),
    })
    flights = pd.DataFrame({
        "Loyalty Number":      rng.choice(np.arange(1, n + 1), size=n * 3),
        "Total Flights":       rng.integers(0, 80, n * 3),
        "Distance":            rng.integers(0, 100_000, n * 3),
        "Points Accumulated":  rng.integers(0, 50_000, n * 3),
        "Points Redeemed":     rng.integers(0, 5_000, n * 3),
    })

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    loyalty_path = raw / "Customer Loyalty History.csv"
    flights_path = raw / "Customer Flight Activity.csv"
    loyalty.to_csv(loyalty_path, index=False)
    flights.to_csv(flights_path, index=False)

    monkeypatch.setattr(data_access, "LOCAL_LOYALTY_CSV", str(loyalty_path))
    monkeypatch.setattr(data_access, "LOCAL_FLIGHT_CSV", str(flights_path))
    data_access._load_local_frame.cache_clear()
    yield
    data_access._load_local_frame.cache_clear()
