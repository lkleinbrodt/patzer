"""
R2 push/pull helpers. All functions are no-ops if R2 env vars aren't set.

R2 mirrors local path structure:
  data/prepared/train.bin       ↔  R2: data/prepared/train.bin
  checkpoints/patzer_v0/ckpt.pt ↔  R2: checkpoints/patzer_v0/ckpt.pt

Usage:
  python r2.py push data/prepared          # upload all files in a local dir
  python r2.py pull data/prepared          # download all objects under that prefix
  python r2.py push checkpoints/patzer_v0  # upload a checkpoint dir
"""

import os
import sys
from pathlib import Path


def _client():
    import boto3
    from dotenv import load_dotenv
    load_dotenv()
    endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
    key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
    if not all([endpoint, key_id, secret]) or "<YOUR_CLOUDFLARE" in endpoint:
        return None, None
    bucket = os.environ.get("R2_BUCKET", "patzer").strip()
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        region_name="auto",
    )
    return client, bucket


def push_file(local_path: str | Path, r2_key: str | None = None) -> bool:
    """Upload a single file. r2_key defaults to the local path string."""
    client, bucket = _client()
    if client is None:
        return False
    local_path = Path(local_path)
    if r2_key is None:
        r2_key = str(local_path)
    print(f"[r2] pushing {local_path} → {r2_key}")
    client.upload_file(str(local_path), bucket, r2_key)
    return True


def pull_file(r2_key: str, local_path: str | Path | None = None) -> bool:
    """Download a single file. local_path defaults to r2_key."""
    client, bucket = _client()
    if client is None:
        return False
    if local_path is None:
        local_path = Path(r2_key)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[r2] pulling {r2_key} → {local_path}")
    client.download_file(bucket, r2_key, str(local_path))
    return True


def push_dir(local_dir: str | Path, r2_prefix: str | None = None) -> bool:
    """Upload all files under local_dir, preserving relative structure."""
    client, bucket = _client()
    if client is None:
        return False
    local_dir = Path(local_dir)
    if r2_prefix is None:
        r2_prefix = str(local_dir)
    files = list(local_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    for f in files:
        rel = f.relative_to(local_dir)
        key = f"{r2_prefix}/{rel}".replace("\\", "/")
        push_file(f, key)
    print(f"[r2] pushed {len(files)} files from {local_dir}")
    return True


def pull_dir(r2_prefix: str, local_dir: str | Path | None = None) -> bool:
    """Download all objects under r2_prefix into local_dir."""
    client, bucket = _client()
    if client is None:
        return False
    if local_dir is None:
        local_dir = Path(r2_prefix)
    local_dir = Path(local_dir)
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=r2_prefix)
    count = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(r2_prefix):].lstrip("/")
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[r2] pulling {key} → {local_path}")
            client.download_file(bucket, key, str(local_path))
            count += 1
    print(f"[r2] pulled {count} files into {local_dir}")
    return count > 0


def checkpoint_exists(r2_key: str) -> bool:
    """Return True if the given key exists in R2."""
    client, bucket = _client()
    if client is None:
        return False
    try:
        client.head_object(Bucket=bucket, Key=r2_key)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("push", "pull"):
        print("usage: python r2.py push <local_dir>")
        print("       python r2.py pull <r2_prefix> [local_dir]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "push":
        push_dir(sys.argv[2])
    elif cmd == "pull":
        local = sys.argv[3] if len(sys.argv) > 3 else None
        pull_dir(sys.argv[2], local)
