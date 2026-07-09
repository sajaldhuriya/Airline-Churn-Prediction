"""
Smoke tests for the Docker entrypoint logic.

We don't have a Docker daemon in CI, so we can't actually `docker build`.
Instead we test the things that would go wrong silently at runtime if
the Dockerfile is wrong:

  1. The `CMD` parses into a real uvicorn command with a resolvable
     module path.
  2. The `HEALTHCHECK` URL matches a known route on the app.
  3. The module path is importable and exposes the expected symbol.

If the Dockerfile changes the entrypoint (e.g. to `gunicorn` or
`uvicorn --workers 4`), these tests catch it and force an explicit
update here — which is what we want, because the deploy story depends
on the entrypoint shape.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "infrastructure"
DOCKERFILE_API = INFRA / "Dockerfile"


def _parse_cmd(content: str) -> list[str]:
    """Extract the CMD instruction as a list of args.

    Supports the two valid Dockerfile forms:
        CMD ["a", "b", "c"]
        CMD a b c
    We don't handle the shell form's shell parsing (it's not worth
    doing — exec form is what's used here).
    """
    m = re.search(r'^CMD\s+(.+?)(?=\n(?:[A-Z]|\Z))',
                  content, re.MULTILINE | re.DOTALL)
    assert m, "Dockerfile should declare a CMD"
    raw = m.group(1).strip()
    if raw.startswith("["):
        # JSON-style: ['uvicorn', 'src.api.app:app', ...]
        import json
        return json.loads(raw)
    # Shell form: 'python app.py' → ['python', 'app.py']
    return raw.split()


def _parse_healthcheck(content: str) -> str | None:
    """Pull the URL out of `urllib.request.urlopen('http://...')`."""
    m = re.search(r"urlopen\(\s*['\"]([^'\"]+)['\"]", content)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# API Dockerfile
# ---------------------------------------------------------------------------
def test_api_dockerfile_cmd_uses_uvicorn():
    content = DOCKERFILE_API.read_text()
    cmd = _parse_cmd(content)
    assert cmd[0] == "uvicorn", f"expected uvicorn, got: {cmd}"


def test_api_dockerfile_cmd_module_path_resolves():
    """src.api.app:app — verify the module imports and exposes `app`."""
    content = DOCKERFILE_API.read_text()
    cmd = _parse_cmd(content)
    # Find the second arg (the module:app spec)
    module_spec = next((a for a in cmd if ":" in a), None)
    assert module_spec, f"expected a module:app spec, got: {cmd}"
    module_name, app_name = module_spec.split(":", 1)
    # We can resolve the module from the project root.
    sys.path.insert(0, str(ROOT))
    mod = __import__(module_name, fromlist=[app_name])
    assert hasattr(mod, app_name), (
        f"{module_name!r} does not expose {app_name!r}"
    )


def test_api_dockerfile_healthcheck_url_is_reachable_route():
    """The HEALTHCHECK URL must match a route the app actually serves."""
    content = DOCKERFILE_API.read_text()
    url = _parse_healthcheck(content)
    assert url is not None, "Dockerfile should declare a HEALTHCHECK URL"
    path = url.split("://", 1)[-1].split("/", 1)[1] if "://" in url else url
    # The path should be /health or / (both are valid liveness probes)
    assert path in ("health", ""), (
        f"HEALTHCHECK should target a liveness route, got: {path!r}"
    )


def test_api_health_endpoint_responds_200():
    """Spin up the app in a background uvicorn and hit /health. This
    exercises the full start-to-ready path the Docker container
    follows on `docker run`."""
    import socket
    import subprocess

    # Pick a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Poll for readiness (up to 15s — the lifespan loads MLflow + a
        # full Pipeline, so it's slow on first run).
        deadline = time.time() + 15
        last_err = None
        while time.time() < deadline:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1)
                if r.status_code == 200:
                    # The process is up — that's the liveness contract.
                    # status=ok means the champion is loaded; degraded
                    # means the process is up but the model wasn't found
                    # in MLflow (e.g. fresh checkout, no champion run).
                    # Both are healthy from a liveness-probe standpoint.
                    body = r.json()
                    assert body.get("status") in ("ok", "degraded"), body
                    return
            except requests.RequestException as e:
                last_err = e
            time.sleep(0.5)
        pytest.fail(
            f"uvicorn did not become healthy on :{port} within 15s. "
            f"last_err={last_err}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Dashboard Dockerfile
# ---------------------------------------------------------------------------
def test_dashboard_dockerfile_cmd_runs_flask_app():
    """The dashboard image's CMD should be `python .../app.py`."""
    dash = (INFRA / "Dockerfile.dashboard").read_text()
    cmd = _parse_cmd(dash)
    assert "python" in cmd
    # The last arg should be a script that ends with app.py
    last = cmd[-1]
    assert last.endswith("app.py"), f"expected .../app.py, got: {last}"


def test_dashboard_dockerfile_healthcheck_uses_overview_route():
    """The dashboard's healthcheck should hit /api/overview (the most
    data-rich route — a 200 there means the data layer is alive)."""
    dash = (INFRA / "Dockerfile.dashboard").read_text()
    url = _parse_healthcheck(dash)
    assert url is not None
    assert "/api/overview" in url, (
        f"dashboard healthcheck should target /api/overview, got: {url}"
    )
