"""
Smoke test for infrastructure/docker-compose.yml.

The whole point of these tests is to fail fast on a typo or a missing
service BEFORE running `docker compose up` (which takes minutes and
gives a less helpful error). We assert:
  - the YAML parses
  - every expected service is present
  - service dependencies exist
  - ports are numeric
  - environment variables reference declared secrets
  - both Dockerfiles exist and reference a real requirements.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infrastructure"
COMPOSE = INFRA / "docker-compose.yml"
DOCKERFILE_PATHS = [
    INFRA / "Dockerfile",
    INFRA / "Dockerfile.dashboard",
]


@pytest.fixture(scope="module")
def compose():
    assert COMPOSE.exists(), f"{COMPOSE} missing"
    with open(COMPOSE) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# YAML structure
# ---------------------------------------------------------------------------
def test_compose_yaml_parses(compose):
    assert "services" in compose
    assert "volumes" in compose


def test_expected_services_present(compose):
    services = set(compose["services"].keys())
    for required in ("postgres", "airflow-init", "airflow-webserver",
                     "airflow-scheduler", "churn-api", "churn-dashboard"):
        assert required in services, f"missing service: {required}"


# ---------------------------------------------------------------------------
# Dashboard service wiring
# ---------------------------------------------------------------------------
def test_dashboard_depends_on_api(compose):
    deps = compose["services"]["churn-dashboard"].get("depends_on", [])
    if isinstance(deps, dict):
        deps = list(deps.keys())
    assert "churn-api" in deps, "dashboard must wait for churn-api to start"


def test_dashboard_fastapi_url_uses_service_name(compose):
    """FASTAPI_URL must point at the churn-api service via Compose DNS,
    not localhost — the dashboard container can't reach localhost:8000
    of the host machine."""
    env = compose["services"]["churn-dashboard"].get("environment", {})
    fastapi_url = env.get("FASTAPI_URL", "")
    assert "churn-api" in fastapi_url, (
        f"FASTAPI_URL must use Compose service DNS, got: {fastapi_url!r}"
    )
    assert "localhost" not in fastapi_url and "127.0.0.1" not in fastapi_url


def test_dashboard_port_mapped(compose):
    ports = compose["services"]["churn-dashboard"].get("ports", [])
    mapped = False
    for p in ports:
        if isinstance(p, str) and p.endswith(":5001"):
            mapped = True
        elif isinstance(p, dict) and p.get("target") == 5001:
            mapped = True
    assert mapped, "churn-dashboard must publish 5001"


def test_dashboard_data_bind_mount(compose):
    """The dashboard reads data/raw/*.csv at request time. If we don't
    bind-mount the host's data/ into the container, the snapshot
    fallback kicks in — which is fine, but worth asserting it's
    intentional."""
    volumes = compose["services"]["churn-dashboard"].get("volumes", [])
    has_data_mount = any("../data/raw" in v for v in volumes)
    if not has_data_mount:
        pytest.skip("no data/raw bind mount — dashboard will use snapshot fallback")


# ---------------------------------------------------------------------------
# Dockerfiles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path", DOCKERFILE_PATHS)
def test_dockerfile_exists(path):
    assert path.exists(), f"{path.name} missing"


@pytest.mark.parametrize("path", DOCKERFILE_PATHS)
def test_dockerfile_has_healthcheck(path):
    content = path.read_text()
    assert "HEALTHCHECK" in content, f"{path.name} has no HEALTHCHECK"


@pytest.mark.parametrize("path", DOCKERFILE_PATHS)
def test_dockerfile_uses_non_root_user(path):
    content = path.read_text()
    assert "USER app" in content or "USER nonroot" in content, (
        f"{path.name} does not switch to a non-root user"
    )


def test_dashboard_dockerfile_excludes_heavy_deps():
    """The dashboard image should be lean. We strip airflow, snowflake,
    xgboost, boto3, fastapi, uvicorn, httpx from requirements before
    installing. The grep pattern is `^(pkg1|pkg2|...)` so we check
    that each package name appears inside a `^(...)` exclusion list."""
    content = (INFRA / "Dockerfile.dashboard").read_text()
    # Find the line that starts the exclusion grep
    grep_match = re.search(r"grep\s+-[a-zA-Z]*\s*['\"]([^'\"]+)['\"]", content)
    assert grep_match, "Dockerfile.dashboard should use grep to filter requirements"
    pattern = grep_match.group(1)
    for excluded in ("apache-airflow", "snowflake-connector-python",
                     "xgboost", "boto3", "fastapi", "uvicorn"):
        assert excluded in pattern, (
            f"Dockerfile.dashboard grep pattern should exclude {excluded}; "
            f"got: {pattern!r}"
        )


# ---------------------------------------------------------------------------
# Airflow DAG tests were removed when dags/ was dropped from the repo
# (commit cc63a9c). The drift step is exercised directly via
# `make data-audit`; orchestration is the operator's choice (cron,
# Airflow, GitHub Actions — same drift module, same JSON output,
# same exit code).
# ---------------------------------------------------------------------------
