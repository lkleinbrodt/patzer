"""
Migrate older R2 checkpoint keys to the new naming scheme.

Old (historical):
  ckpt_best.pt
  ckpt_<iter>.pt   e.g. ckpt_010000.pt, ckpt_065000.pt (numeric suffix only)

New:
  weights_best.pt
  weights_iter_<iter>.pt   e.g. weights_iter_010000.pt (suffix preserved as-is)

`ckpt.pt` (latest resume blob) is not renamed — only `ckpt_best.pt` and `ckpt_<digits>.pt`.

By default we do not delete old keys; we server-side copy within the bucket.
Copy is skipped when the destination already exists unless you pass --force.
If you pass --move, we delete the source key after verifying the destination exists.

Usage:
  python -m patzer.migrate_r2_checkpoint_names --prefix checkpoints/patzer_v3
  python -m patzer.migrate_r2_checkpoint_names --prefix checkpoints/patzer_v2
  python -m patzer.migrate_r2_checkpoint_names --prefix checkpoints/patzer_v2 --force
  python -m patzer.migrate_r2_checkpoint_names --prefix checkpoints/patzer_v3 --move
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import r2


def _list_pt_keys(prefix: str) -> list[str]:
    client, bucket = r2._client()  # type: ignore[attr-defined]
    if client is None:
        raise SystemExit("R2 not configured — check .env credentials")
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".pt"):
                keys.append(key)
    return keys


def _delete_key(key: str) -> bool:
    client, bucket = r2._client()  # type: ignore[attr-defined]
    if client is None:
        return False
    try:
        client.delete_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="R2 prefix e.g. checkpoints/patzer_v3")
    ap.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    ap.add_argument(
        "--move",
        action="store_true",
        help="Delete source keys after successful copy + destination verification",
    )
    args = ap.parse_args()

    prefix = args.prefix.strip().lstrip("/")
    keys = _list_pt_keys(prefix)
    if not keys:
        print(f"[migrate] no .pt objects under {prefix!r}")
        return

    copies: list[tuple[str, str]] = []
    for k in keys:
        name = Path(k).name
        if name == "ckpt_best.pt":
            copies.append((k, f"{Path(k).parent.as_posix()}/weights_best.pt"))
        elif name.startswith("ckpt_") and name.endswith(".pt"):
            tail = name.removeprefix("ckpt_").removesuffix(".pt")
            if tail.isdigit():
                copies.append((k, f"{Path(k).parent.as_posix()}/weights_iter_{tail}.pt"))

    if not copies:
        print(f"[migrate] nothing to copy under {prefix!r}")
        return

    ok = 0
    for src, dst in copies:
        did = r2.copy_object(src, dst, overwrite=args.force)
        if not did:
            continue
        # Verify destination exists before deleting source.
        if args.move:
            if not r2.checkpoint_exists(dst):
                print(f"[migrate] move skipped; destination missing after copy: {dst}")
                continue
            if not _delete_key(src):
                print(f"[migrate] move warning: failed to delete source: {src}")
                continue
        ok += 1

    verb = "moved" if args.move else "copied"
    print(f"[migrate] {verb} {ok}/{len(copies)} objects")


if __name__ == "__main__":
    main()

