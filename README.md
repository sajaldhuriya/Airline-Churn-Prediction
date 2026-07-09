# AeroRetain AI — Airline Loyalty Churn Prediction

A focused machine learning system that predicts whether a loyalty-program
member will stop flying with the airline, ships the prediction through a
FastAPI service, and shows it to managers through a Flask dashboard.

## What this project is

A self-contained example of an end-to-end churn-prediction product:

- **Data** — joins `Customer Loyalty History.csv` with
  `Customer Flight Activity.csv` (or pulls the same table from Snowflake
  in the production path).
- **Model** — `Pipeline(StandardScaler → SVC rbf, C=1.0, class_weight=balanced)`
  on an engagement-only feature set (4 numerics + 15 dummies). Champion
  selected by holdout recall with a bootstrap 95% CI. Wrapped in a
  Pipeline so the inference layer applies the same scaling the model
  was trained on.
- **Service** — FastAPI exposes `/predict` with strict Pydantic
  validation (one-of-N for every categorical). A lifespan handler
  loads the champion from MLflow.
- **UI** — Flask dashboard with four pages: KPIs, BCG strategy,
  statistical battery, and a single-customer scoring form. KPI tiles
  show bootstrap confidence intervals, not point estimates. The stats
  page now surfaces **per-segment recall** so weak sub-populations
  (2018-Promotion, Star card) are visible without a notebook.
- **Quality** — a standalone Data Quality Audit script runs KS tests on
  numerics and chi²/Cramér's V on categoricals. Wire it into any
  scheduler and treat `drift_report.json` as a gate.

## Architecture

```
┌──────────────┐    ┌────────────────┐    ┌────────────────┐    ┌──────────────┐
│  data/raw/   │    │  src/          │    │  MLflow        │    │  FastAPI     │
│  *.csv  ─────┼───▶│  data_pipeline │───▶│  (SQLite)      │───▶│  /predict    │
│  or          │    │  models        │    │  champion run  │    │  /health     │
│  Snowflake   │    │                │    │                │    └──────┬───────┘
└──────────────┘    └────────────────┘    └────────────────┘           │
       │                                            │                  ▼
       │                                            │         ┌──────────────┐
       │                                            │         │ Flask UI     │
       │                                            │         │ manager view │
       │                                            │         └──────────────┘
       │                                            │
       │                                  ┌─────────┴────────┐
       │                                  │  permutation_    │
       │                                  │  importances.json│
       │                                  │  + feature_names │
       │                                  │  .txt sidecar    │
       │                                  └──────────────────┘
       │
       └──▶ statistical battery (notebooks/) ──▶ effect sizes + CIs
            machine learning (notebooks/)  ──▶ champion selection
            drift.py                      ──▶ drift_report.json
```

| Layer | Code | Purpose |
|---|---|---|
| Ingest | `src/data_pipeline/s3_upload.py`, `pull_snowflake.py` | Land raw CSVs in S3, then pull and clean from Snowflake. Local CSVs are the dev fallback. |
| Train | `src/models/train_svc.py` | Fit the SVC pipeline and log to MLflow. |
| Tune | `src/models/tune_arena.py` | Sweep 4 model families, optimize for recall. |
| Stats | `src/models/statistical_tests.py` | Welch, Mann-Whitney, Levene, Chi², Cramér's V, bootstrap CIs. |
| Audit | `src/models/drift.py` | Data quality audit — KS + chi² between baseline and recent. |
| Serve | `src/api/app.py` | FastAPI; loads the Pipeline; serves `/predict` and `/health`. |
| UI | `business_strategy/flask_dashboard/` | Manager dashboard, calls FastAPI. |
| Container | `infrastructure/` | Multi-stage Dockerfile + Compose. |

## Quick start

```bash
# 1. Configure environment
cp .env.example .env
# fill in only the vars you have. Local CSVs work without Snowflake.
make env                # validates

# 2. Install
make install

# 3. Train (logs the champion to MLflow)
make retrain

# 4. Serve
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
# in another shell
python business_strategy/flask_dashboard/app.py
# open http://127.0.0.1:5001

# Swagger UI (interactive docs):
# open http://127.0.0.1:8000/docs
```

## API endpoints

Once `uvicorn src.api.app:app --port 8000` is running:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/`              | Status, active champion name, validation recall, feature count. |
| `GET`  | `/health`        | `{"status": "ok"}` or `{"status": "degraded", ...}`. |
| `POST` | `/predict`       | 19-field customer profile → `{churn_prediction, probability, served_by}`. |
| `GET`  | `/docs`          | Swagger UI (auto-generated from the Pydantic schema). |
| `GET`  | `/redoc`         | ReDoc. |
| `GET`  | `/openapi.json`  | Raw OpenAPI schema. |

## Manager playbook — what to do when the model flags a customer

The model is a triage tool, not a decision-maker. It tells a manager
**which customers to spend attention on**; the manager decides the
intervention. The strategies below are ordered by how often they
work, based on the engagement-vs-wealth story the stats battery
established (LIFETIME_FLIGHTS Cohen's d ≈ 1.6 against churn, SALARY
and CLV ≈ 0).

### 1. Re-ignite engagement (the primary lever)

Engagement — flights, distance, points earned, points redeemed — is
the only thing in the data that actually predicts churn. A customer
who has flown recently and redeemed points recently is retained even
if their salary is low; a customer who hasn't flown in a year and
hasn't redeemed anything is leaving regardless of wealth.

**Concrete actions for a flagged customer:**

- **Trigger a "second-flight moment"** — within 30 days of signup, a
  customer with one flight is the most at-risk. Send a one-time
  discount on a second flight.
- **Drive redemption, not earning** — churners redeem far fewer
  points than retainers. A targeted "you have 12,000 points — book
  your next trip on us" beats generic earning campaigns.
- **Personalise by route history** — the LIFETIME_DISTANCE feature is
  the second-strongest predictor (|d| ≈ 1.5). Customers who flew
  long-haul and then stopped are usually responding to a
  schedule/route change, not a price change.

### 2. Don't chase the wrong signal

The stats battery proved SALARY and CLV are statistically
indistinguishable from churn. **Money incentives targeted at
high-CLV-but-disengaged customers are wasted spend.** The model
already accounts for CLV implicitly through lifetime flights, so a
high-CLV customer with no recent engagement is exactly the kind of
false-positive a manager should ignore.

**Concrete actions:**

- Do not send 20%-off coupons to a Star-card holder who flies
  monthly. They will fly monthly with or without the coupon.
- Do not invest in concierge service for a high-CLV customer who
  is also flying monthly — they are not at risk.
- Do invest in concierge service for a customer whose CLV is high
  *because* of past flying, but who has stopped flying in the last
  6 months. That is the only segment where the spend is justified.

### 3. Watch the weak segments

Per-segment recall (notebook section 7, now visible on the
`/stats` page) is the right way to decide which cohorts need their
own model. The notebook flagged two:

| Segment | Recall on champion | What to do |
|---|---|---|
| **2018-Promotion cohort** | 0.45 (very weak) | Acquired during a one-time promo; engagement signature doesn't match the population. Don't apply the global model — they need their own feature engineering, or a separate model trained on promotion-only data. |
| **Star card** | 0.80 (just below target) | The largest segment by count. A 5-point recall improvement here is more absolute churners caught than a 50-point improvement in a tiny segment. Worth a focused retraining pass. |

The `Star` and `Aurora` cards have similar churn rates (~12%) but
Star has the most customers, so improving Star's recall is the
highest-leverage move in the system.

### 4. Use the bootstrap CI, not the point estimate

The macro churn rate on the dashboard is shown as a 95% bootstrap CI,
not a point. When the manager-facing tiles say "16.4% (15.8%–17.0%)",
the right read is "we expect churn somewhere in this band." If
next quarter's number lands outside the band, something real has
changed (route cancellations, fuel-price shock, competitor entry)
and the model needs a retrain — not a discount campaign.

## Data Quality Audit

`make data-audit` runs `src/models/drift.py`, which compares a recent
feature distribution against a training-time baseline:

- **Numeric features** (LIFETIME_FLIGHTS, LIFETIME_DISTANCE, …): two-sample
  Kolmogorov-Smirnov test.
- **Categorical features** (the dummies from GENDER, EDUCATION, …):
  chi-square test on value proportions with Cramér's V as the effect
  size.

A feature is flagged as drifting when **both** the p-value is below 0.05
**and** the effect size is at least "small" (KS statistic or Cramér's V
≥ 0.10). Either alone is ignored — the n=16K sample size means a
negligible effect will still produce a tiny p-value, which is the
statistical-power trap the stats battery also warns about.

The report is written to `drift_report.json`. The script exits
non-zero if anything is drifting, so it can be wired into any
scheduler (cron, GitHub Actions, Airflow, etc.) as a gate.

## Tests

The `tests/` folder is kept on disk for local development and is
**gitignored** — it does not reach GitHub. Run the suite with:

```bash
make test
```

It's hermetic: no Snowflake, no MLflow state, no live model needed
(the `synthetic_csvs` fixture in `conftest.py` generates a fake
dataset when the real CSVs are absent).

## Key design decisions

1. **The scaler is part of the model.** `Pipeline(StandardScaler →
   SVC rbf)` is logged to MLflow as a single artifact, so the API
   applies the same transform that was fit at training. Without this,
   raw `LIFETIME_FLIGHTS` would dominate the decision boundary.
2. **We optimize for recall, not accuracy.** A missed churner (FN)
   costs more than a false positive (FP) — sending a discount to a
   retained user is cheap; losing a high-CLV customer is expensive.
3. **The schema mirrors the sidecar.** The Pydantic schema in
   `src/api/app.py` is exactly the 19 columns in
   `models/feature_names.txt`. The API reads the sidecar at startup
   and rejects any input that doesn't match.
4. **Drivers on the dashboard come from the model, not a hand-curated
   dict.** `get_model_drivers()` reads the `permutation_importances.json`
   artifact the training script logs, ranks features by importance,
   and returns the top-4 as percentages summing to 100. The rbf
   kernel has no `coef_`, so permutation importance is the only
   model-derived signal.
5. **Per-segment recall is a first-class surface.** The
   `/api/segments` endpoint and the stats page both surface
   per-segment recall computed live from the champion on the local
   CSV. Weak segments are flagged with a "weak" or "below 75%" badge
   so the manager doesn't have to read a notebook.
6. **Headline numbers carry uncertainty.** The churn-rate KPI tile
   shows a 95% bootstrap CI so a stakeholder sees the *range*, not a
   point estimate.
7. **Drift is decoupled from the orchestrator.** `drift.py` is a
   standalone CLI — it just reads two DataFrames and writes a JSON
   report. Wire it into whatever scheduler you already have.

## Repository layout

```
.
├── configs/                 # YAML model hyperparameters
├── data/raw/                # Local raw CSVs (gitignored)
├── infrastructure/          # Dockerfile + docker-compose.yml
├── models/                  # Saved champion pipeline + feature names sidecar
├── notebooks/               # Exploratory notebooks (statistical battery + ML arena)
├── business_strategy/       # Flask UI + data-access layer
├── scripts/                 # Dev helpers (env check, dryrun, retrain)
├── src/
│   ├── api/app.py           # FastAPI service
│   ├── data_pipeline/       # Snowflake + S3
│   └── models/              # Train, tune, stats, drift
├── tests/                   # Hermetic test suite (gitignored)
├── Makefile
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
