# Airline Loyalty Churn — Makefile shortcuts.
# All targets are idempotent and run from the project root.

PY ?= python
PIP ?= pip
DOCKER ?= docker

.PHONY: install test retrain tune stats data-audit docker-build env lint clean

install:
	$(PIP) install -r requirements.txt

# Fast: hermetic test suite. tests/ is gitignored — runs only locally.
test:
	$(PY) -m pytest tests/ -v --maxfail=1

# Run the full retrain: train the SVC champion, log to MLflow.
retrain:
	$(PY) src/models/train_svc.py

# Hyperparameter sweep across all 4 model families.
tune:
	$(PY) src/models/tune_arena.py

# Statistical test battery (Welch, Mann-Whitney, Levene, Chi², Cramér's V).
stats:
	$(PY) src/models/statistical_tests.py

# Data quality audit — KS test on numerics + chi²/Cramér's V on
# categoricals. Writes drift_report.json. Exit non-zero if any
# feature is drifting.
data-audit:
	$(PY) src/models/drift.py --out drift_report.json

docker-build:
	$(DOCKER) build -t airline_churn_api -f infrastructure/Dockerfile .

# Verify .env is configured. Fails loudly if any required var is missing.
env:
	@$(PY) scripts/check_env.py

lint:
	$(PY) -m pyflakes src/ tests/ || true

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
	find . -name "*.pyc" -delete
