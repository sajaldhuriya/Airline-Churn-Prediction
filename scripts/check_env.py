"""
Validates that all required env vars are set. Run via `make env`.
Exits non-zero with a clear list of missing keys.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "SNOWFLAKE_USER": "Snowflake username",
    "SNOWFLAKE_PASSWORD": "Snowflake password (use OAuth/SSO if possible)",
    "SNOWFLAKE_ACCOUNT": "Snowflake account locator, e.g. xy12345.us-east-1",
}

OPTIONAL = {
    "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
    "SNOWFLAKE_DATABASE": "AIRLINE_LOYALTY_DB",
    "SNOWFLAKE_SCHEMA": "RAW_DATA",
    "AWS_ACCESS_KEY_ID": "AWS access key for S3 upload",
    "AWS_SECRET_ACCESS_KEY": "AWS secret key for S3 upload",
    "AWS_REGION": "us-east-1",
    "AWS_S3_BUCKET_NAME": "S3 bucket for raw data landing",
    "FASTAPI_URL": "http://127.0.0.1:8000/predict",
    "FLASK_DEBUG": "0",
}


def main() -> int:
    env_path = ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)

    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        print("❌ Missing required env vars:")
        for k in missing:
            print(f"   - {k}  ({REQUIRED[k]})")
        print(f"\nCopy .env.example to .env and fill them in.")
        return 1

    print("✅ All required env vars are set.")
    for k, default in OPTIONAL.items():
        v = os.getenv(k)
        marker = "✓" if v else "·"
        print(f"  {marker} {k} = {v if v else f'(default: {default})'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
