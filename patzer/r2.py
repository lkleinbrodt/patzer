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

pull_dir skips files that already exist locally when they match R2 (ETag sidecar),
or when R2 is unreachable (same behavior as is_fresh). Use --force to always
re-download. Pushes write `.r2meta` sidecars after upload so freshness checks work.

The boto3 client is cached with adaptive retries on the underlying botocore session.

Upload design (training loop):
  * Small/medium uploads use **put_object** (single HTTP request). Larger files
    use **multipart** uploads (serial ``upload_part`` calls — no S3Transfer /
    nested thread pools). Threshold defaults below the typical 5 GiB single-PUT cap.
  * ``push_async`` uses one background worker so large checkpoint uploads do not
    block GPU steps for minutes, but ``flush_r2_uploads()`` drains the queue on
    normal training exit and from atexit so R2 does not lag arbitrarily behind.
  * If ``ThreadPoolExecutor.submit`` fails (shutdown races), the upload runs
    synchronously in the caller — never silently dropped.
"""

import atexit
import itertools
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from botocore.config import Config

# Single-worker executor: uploads queue up and never run concurrently, so
# there's no risk of two in-flight uploads stomping each other on R2.
_executor = ThreadPoolExecutor(max_workers=1)
# Drain all pending uploads on normal exit (including KeyboardInterrupt / Ctrl-C).
atexit.register(_executor.shutdown, wait=True)

# Monotonic counter for unique temp file names — prevents collisions when the
# same local_path is enqueued again before a prior upload of that path finishes.
_upload_counter = itertools.count()

# Approximate queue depth (increment when a job is submitted, decrement when it finishes).
_pending_lock = threading.Lock()
_pending_async_uploads = 0
_WARN_QUEUE_DEPTH = 5

_client_lock = threading.Lock()
_client_cache: tuple[object, str] | None = None

# Single PutObject bodies are limited (S3/R2: 5 GiB). Multipart parts must be
# ≥ 5 MiB except the last. Threshold/part size env vars are read in ``_upload_one_file``
# after ``_client()`` runs ``load_dotenv()``.
_MP_MIN_PART = 5 * 1024 * 1024
_LARGE_SINGLE_PUT_NOTE = 100 * 1024 * 1024  # log once: single PUT has no incremental progress


def _upload_one_file(client, bucket: str, r2_key: str, local_path: Path, *, quiet: bool) -> dict:
    """Upload ``local_path``; return boto response dict with ``ETag`` if present."""
    threshold = int(
        os.environ.get("R2_SINGLE_PUT_THRESHOLD_BYTES", str(4 * 1024 * 1024 * 1024))
    )
    part_max = max(
        int(os.environ.get("R2_MULTIPART_PART_BYTES", str(64 * 1024 * 1024))),
        _MP_MIN_PART,
    )

    size = local_path.stat().st_size
    if size <= threshold:
        if not quiet and size >= _LARGE_SINGLE_PUT_NOTE:
            print(
                f"[r2] put_object {r2_key} ({size} bytes — single request, waits until upload finishes)",
                flush=True,
            )
        with open(local_path, "rb") as f:
            return client.put_object(Bucket=bucket, Key=r2_key, Body=f)

    est_parts = max(1, (size + part_max - 1) // part_max)
    log_every = max(1, est_parts // 25)  # ~25 progress lines over the whole upload
    if not quiet:
        print(
            f"[r2] multipart {r2_key} starting ({size} bytes, ~{est_parts} parts @ ≤{part_max // (1024 * 1024)} MiB)",
            flush=True,
        )

    resp = client.create_multipart_upload(Bucket=bucket, Key=r2_key)
    upload_id = resp["UploadId"]
    parts: list[dict] = []
    try:
        with open(local_path, "rb") as f:
            part_num = 1
            bytes_done = 0
            remaining = size

            while remaining > 0:
                if remaining <= part_max:
                    to_read = remaining
                else:
                    to_read = part_max
                    tail = remaining - to_read
                    if 0 < tail < _MP_MIN_PART:
                        to_read = remaining - _MP_MIN_PART

                chunk = f.read(to_read)
                if len(chunk) != to_read:
                    raise RuntimeError(
                        f"short read uploading {local_path}: got {len(chunk)} bytes, expected {to_read}"
                    )

                up = client.upload_part(
                    Bucket=bucket,
                    Key=r2_key,
                    PartNumber=part_num,
                    UploadId=upload_id,
                    Body=chunk,
                )
                parts.append({"ETag": up["ETag"], "PartNumber": part_num})
                part_num += 1
                bytes_done += len(chunk)
                remaining -= len(chunk)

                done_parts = len(parts)
                if not quiet and (
                    done_parts == 1
                    or done_parts % log_every == 0
                    or remaining == 0
                ):
                    pct = min(100, 100 * bytes_done / size)
                    print(
                        f"[r2] multipart {r2_key} {pct:.0f}% "
                        f"(part {done_parts}/{est_parts}+, {bytes_done}/{size} bytes)",
                        flush=True,
                    )

        if not parts:
            raise RuntimeError(f"multipart upload produced no parts: {local_path}")

        return client.complete_multipart_upload(
            Bucket=bucket,
            Key=r2_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": sorted(parts, key=lambda p: p["PartNumber"])},
        )
    except BaseException:
        try:
            client.abort_multipart_upload(Bucket=bucket, Key=r2_key, UploadId=upload_id)
        except Exception:
            pass
        raise


def _client():
    """Return a cached (boto3 client, bucket) or (None, None) if R2 is not configured."""
    global _client_cache
    with _client_lock:
        if _client_cache is not None:
            return _client_cache
        import boto3
        from dotenv import load_dotenv

        load_dotenv()
        endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
        key_id = os.environ.get("R2_ACCESS_KEY_ID", "").strip()
        secret = os.environ.get("R2_SECRET_ACCESS_KEY", "").strip()
        if not all([endpoint, key_id, secret]) or "<YOUR_CLOUDFLARE" in endpoint:
            _client_cache = (None, None)
            return None, None
        bucket = os.environ.get("R2_BUCKET", "patzer").strip()
        cfg = Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            connect_timeout=60,
            read_timeout=300,
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name="auto",
            config=cfg,
        )
        _client_cache = (client, bucket)
        return _client_cache


def push_file(local_path: str | Path, r2_key: str | None = None) -> bool:
    """Upload a single file. r2_key defaults to the local path string."""
    client, bucket = _client()
    if client is None:
        return False
    local_path = Path(local_path)
    if r2_key is None:
        r2_key = str(local_path)
    print(f"[r2] pushing {local_path} → {r2_key}", flush=True)
    resp = _upload_one_file(client, bucket, r2_key, local_path, quiet=False)
    etag = (resp.get("ETag") or "").strip('"')
    if etag:
        _write_sidecar(local_path, etag)
    else:
        _write_sidecar_from_remote(client, bucket, r2_key, local_path)
    return True


def push_file_threadsafe(local_path: str | Path, r2_key: str | None = None) -> bool:
    """Same as ``push_file`` — kept as an explicit name for atexit / signal handlers.

    Uses the same upload path as ``push_file`` (single PUT under threshold, multipart above).
    """
    return push_file(local_path, r2_key)


def flush_r2_uploads(timeout: float | None = None) -> None:
    """Block until all queued ``push_async`` uploads finish.

    Submits a no-op on the single-worker executor so every pending upload
    ahead of it completes first. Call this on normal training shutdown so R2
    is not arbitrarily far behind the local checkpoint dir.
    """
    try:
        fut = _executor.submit(lambda: None)
    except RuntimeError as exc:
        print(f"[r2] flush_r2_uploads: executor closed ({exc})", file=sys.stderr)
        return
    try:
        fut.result(timeout=timeout)
    except Exception as exc:
        print(f"[r2] flush_r2_uploads: {exc}", file=sys.stderr)


def push_async(
    local_path: str | Path,
    r2_key: str | None = None,
    then_copy_to: str | None = None,
) -> None:
    """
    Upload a file in a background thread without blocking the caller.

    Copies local_path to a temporary file first (.uploading sibling), so the
    caller can safely overwrite or delete local_path while the upload runs.
    The temp file is deleted automatically when the upload completes.

    If then_copy_to is given, a server-side R2 copy from r2_key → then_copy_to
    is performed after the upload finishes (no extra data transfer). This is
    used for creating weights_iter_*.pt snapshots safely: the copy must happen
    after the new weights are live on R2, not before.

    Uses the same multipart-aware upload path as ``push_file`` (no ``upload_file`` / S3Transfer).

    Uploads are serialised (max_workers=1). If ``submit`` fails, uploads
    synchronously in the foreground instead of dropping the work.

    Call ``flush_r2_uploads()`` at training shutdown so R2 catches up.

    Falls back to a no-op if R2 is not configured.
    """
    global _pending_async_uploads
    client, bucket = _client()
    if client is None:
        return
    local_path = Path(local_path)
    if r2_key is None:
        r2_key = str(local_path)

    n = next(_upload_counter)
    tmp = local_path.with_name(f"{local_path.stem}.uploading{n}{local_path.suffix}")
    shutil.copy2(local_path, tmp)

    def _work_inner() -> None:
        try:
            print(f"[r2] pushing {local_path} → {r2_key} (async)", flush=True)
            _upload_one_file(client, bucket, r2_key, tmp, quiet=False)
            if then_copy_to:
                copy_object(r2_key, then_copy_to, overwrite=False)
        except Exception as exc:
            print(f"[r2] ERROR uploading {local_path} → {r2_key}: {exc}", file=sys.stderr)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    with _pending_lock:
        _pending_async_uploads += 1
        depth = _pending_async_uploads
    if depth > _WARN_QUEUE_DEPTH:
        print(
            f"[r2] WARNING: {depth} upload(s) queued — R2 may lag behind training "
            f"(slow network or very frequent saves). Consider longer ckpt_save_interval.",
            file=sys.stderr,
        )

    def _job() -> None:
        global _pending_async_uploads
        try:
            _work_inner()
        finally:
            with _pending_lock:
                _pending_async_uploads -= 1

    try:
        _executor.submit(_job)
    except RuntimeError as exc:
        with _pending_lock:
            _pending_async_uploads -= 1
        print(
            f"[r2] async submit failed ({exc}); uploading synchronously in foreground",
            file=sys.stderr,
        )
        _work_inner()


def _sidecar(local_path: Path) -> Path:
    return local_path.with_suffix(local_path.suffix + ".r2meta")


def _read_sidecar(local_path: Path) -> str | None:
    s = _sidecar(local_path)
    return s.read_text().strip() if s.exists() else None


def _write_sidecar(local_path: Path, etag: str) -> None:
    _sidecar(local_path).write_text(etag)


def _remote_etag(client, bucket: str, r2_key: str) -> str | None:
    try:
        resp = client.head_object(Bucket=bucket, Key=r2_key)
        return resp["ETag"].strip('"')
    except Exception:
        return None


def _write_sidecar_from_remote(client, bucket: str, r2_key: str, local_path: Path) -> None:
    etag = _remote_etag(client, bucket, r2_key)
    if etag:
        _write_sidecar(local_path, etag)


def get_etag(r2_key: str) -> str | None:
    """Return the ETag for an R2 object, or None if not found / R2 not configured."""
    client, bucket = _client()
    if client is None:
        return None
    return _remote_etag(client, bucket, r2_key)


def is_fresh(r2_key: str, local_path: Path) -> bool:
    """
    True if the local file matches R2 (via stored ETag sidecar).
    Falls back to True when R2 is unreachable so we don't block offline runs.
    Files with no sidecar are treated as stale only when R2 is reachable.
    """
    if not local_path.exists():
        return False
    local = _read_sidecar(local_path)
    remote = get_etag(r2_key)
    if remote is None:
        return True   # R2 unreachable → assume fresh (offline mode)
    if local is None:
        return False  # R2 online but no sidecar → re-pull to get fresh sidecar
    return local == remote


def list_weights(r2_prefix: str) -> list[str]:
    """List all weights_*.pt keys under r2_prefix, sorted by name."""
    client, bucket = _client()
    if client is None:
        return []
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=r2_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = Path(key).name
            if name.startswith("weights_") and name.endswith(".pt"):
                keys.append(key)
    return sorted(keys)


def pull_file(r2_key: str, local_path: str | Path | None = None, *, skip_existing: bool = False) -> bool:
    """Download a single file and write an ETag sidecar. local_path defaults to r2_key.

    When skip_existing is True, skips the download if the local file is fresh
    (sidecar ETag matches R2); otherwise re-pulls.
    """
    client, bucket = _client()
    if client is None:
        return False
    if local_path is None:
        local_path = Path(r2_key)
    local_path = Path(local_path)
    if skip_existing and local_path.exists():
        if is_fresh(r2_key, local_path):
            print(f"[r2] skipping {r2_key} (fresh local copy)")
            return True
        print(f"[r2] re-pulling {r2_key} (stale or missing sidecar)")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[r2] pulling {r2_key} → {local_path}")
    client.download_file(bucket, r2_key, str(local_path))
    _write_sidecar_from_remote(client, bucket, r2_key, local_path)
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
    """Download all objects under r2_prefix into local_dir.

    With skip_existing (default True), skips files whose local ETag sidecar
    matches R2; re-downloads when missing, stale, or when remote is unreachable
    (is_fresh treats unreachable as skip). Use skip_existing=False to always pull.
    """
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
                if is_fresh(key, local_path):
                    print(f"[r2] skipping {key} (fresh local copy)")
                    skipped += 1
                    continue
                print(f"[r2] re-pulling {key} (stale or missing sidecar)")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[r2] pulling {key} → {local_path}")
            client.download_file(bucket, key, str(local_path))
            _write_sidecar_from_remote(client, bucket, key, local_path)
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
