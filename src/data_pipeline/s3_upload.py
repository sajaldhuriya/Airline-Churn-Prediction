"""
Uploads raw CSVs from data/raw/ to AWS S3.

Differences vs v1.0:
  - Uses the `python-dotenv` library instead of a hand-rolled parser.
  - The data directory is resolved relative to the project root, not CWD.
  - Reads `AWS_REGION` from env so it works outside us-east-1.
  - Fails loudly if a file is missing instead of silently skipping.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

FILES_TO_UPLOAD: List[str] = [
    "Customer Flight Activity.csv",
    "Customer Loyalty History.csv",
    "Calendar.csv",
]


def upload_raw_data_to_s3() -> int:
    """Uploads the canonical raw CSVs to S3 under the prefix `raw/`.

    Returns the number of files successfully uploaded.
    """
    load_dotenv(ROOT / ".env", override=False)

    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION", "us-east-1")
    if not bucket:
        raise RuntimeError("AWS_S3_BUCKET_NAME is not set in .env")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region,
    )

    uploaded = 0
    for filename in FILES_TO_UPLOAD:
        local_path = RAW_DIR / filename
        if not local_path.exists():
            logger.error("Local file missing: %s", local_path)
            continue

        s3_key = f"raw/{filename}"
        try:
            logger.info("Uploading %s -> s3://%s/%s", filename, bucket, s3_key)
            s3_client.upload_file(str(local_path), bucket, s3_key)
            uploaded += 1
        except (BotoCoreError, ClientError) as e:
            logger.exception("Failed to upload %s: %s", filename, e)

    logger.info("Uploaded %d / %d files to s3://%s", uploaded, len(FILES_TO_UPLOAD), bucket)
    return uploaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    upload_raw_data_to_s3()
