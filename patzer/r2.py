"""
R2 push/pull helpers. All functions are no-ops if R2 env vars aren't set.

R2 mirrors local path structure exactly:
  data/prepared/train.bin                      ↔  R2: data/prepared/train.bin
  checkpoints/patzer_v2/ckpt.pt                ↔  R2: latest checkpoint (full, for resume)
  checkpoints/patzer_v2/weights_best.pt        ↔  R2: best val-loss weights (eval/play)
  checkpoints/patzer_v2/weights_iter_050000.pt ↔  R2: training-step snapshot

Usage:
  python r2.py push data/prepared               # upload all files in a local dir
  python r2.py pull checkpoints/patzer_v2       # download entire checkpoint dir
  python r2.py pull checkpoints/patzer_v2/weights_best.pt   # download single file
  python r2.py push checkpoints/patzer_v2       # upload a checkpoint dir
  python r2.py copy src/r2/key dst/r2/key       # server-side copy [--force]

pull_dir skips files that already exist locally (use --force to re-download).
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


def pull_file(r2_key: str, local_path: str | Path | None = None, *, skip_existing: bool = False) -> bool:
    """Download a single file. local_path defaults to r2_key."""
    client, bucket = _client()
    if client is None:
        return False
    if local_path is None:
        local_path = Path(r2_key)
    local_path = Path(local_path)
    if skip_existing and local_path.exists():
        print(f"[r2] skipping {r2_key} (already local)")
        return True
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


def pull_dir(r2_prefix: str, local_dir: str | Path | None = None, *, skip_existing: bool = True) -> bool:
    """Download all objects under r2_prefix into local_dir. Skips existing files by default."""
    client, bucket = _client()
    if client is None:
        return False
    if local_dir is None:
        local_dir = Path(r2_prefix)
    local_dir = Path(local_dir)
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=r2_prefix)
    downloaded = skipped = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(r2_prefix):].lstrip("/")
            local_path = local_dir / rel
            if skip_existing and local_path.exists():
                print(f"[r2] skipping {key} (already local)")
                skipped += 1
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[r2] pulling {key} → {local_path}")
            client.download_file(bucket, key, str(local_path))
            downloaded += 1
    print(f"[r2] pulled {downloaded} files, skipped {skipped} into {local_dir}")
    return downloaded > 0 or skipped > 0


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


def copy_object(src_key: str, dst_key: str, *, overwrite: bool = False) -> bool:
    """
    Server-side copy within the same R2 bucket (no local download).

    src_key / dst_key use forward slashes, e.g. checkpoints/patzer_v1/ckpt.pt
    """
    client, bucket = _client()
    if client is None:
        return False
    src_key = str(src_key).lstrip("/")
    dst_key = str(dst_key).lstrip("/")
    if not overwrite and checkpoint_exists(dst_key):
        print(f"[r2] copy skipped: destination already exists: {dst_key}", file=sys.stderr)
        return False
    print(f"[r2] copying s3://{bucket}/{src_key} → s3://{bucket}/{dst_key}")
    client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
    )
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("push", "pull", "copy"):
        print("usage: python r2.py push <local_dir_or_file>")
        print("       python r2.py pull <r2_prefix_or_key> [local_path] [--force]")
        print("       python r2.py copy <src_r2_key> <dst_r2_key> [--force]")
        sys.exit(1)
    cmd = sys.argv[1]
    force = "--force" in sys.argv
    if cmd == "push":
        push_dir(sys.argv[2])
    elif cmd == "pull":
        target = sys.argv[2]
        local = next((a for a in sys.argv[3:] if not a.startswith("--")), None)
        # If target looks like a file (has an extension), pull a single file
        if Path(target).suffix:
            pull_file(target, local, skip_existing=not force)
        else:
            pull_dir(target, local, skip_existing=not force)
    elif cmd == "copy":
        if len(sys.argv) < 4:
            print("usage: python r2.py copy <src_r2_key> <dst_r2_key> [--force]", file=sys.stderr)
            sys.exit(1)
        ok = copy_object(sys.argv[2], sys.argv[3], overwrite=force)
        sys.exit(0 if ok else 1)
