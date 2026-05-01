#!/usr/bin/env bash
# Sync tokenized prepared data from chewy → this repo.
#
# Remote source: chewy:/home/landon/Projects/patzer/data/prepared/
# Local dest:    data/prepared_<UTC timestamp>/
#
# This pulls the *entire* prepared folder contents (e.g. train.bin/val.bin/meta.json).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TS_UTC="$(date -u +"%Y%m%d_%H%M%SZ")"
DEST="data/prepared_${TS_UTC}"
REMOTE_DEFAULT="chewy:/home/landon/Projects/patzer/data/prepared/"
REMOTE="${1:-$REMOTE_DEFAULT}"

mkdir -p "$DEST"

echo "[transfer] remote: $REMOTE"
echo "[transfer] local:  $DEST"

# Notes:
# - Trailing slashes matter: sync *contents* into DEST.
# - --partial/--append-verify make large transfers resumable.
rsync -avP --partial --append-verify --checksum \
  --exclude '.DS_Store' \
  --exclude '*.tmp' \
  --exclude '*.partial' \
  "$REMOTE" \
  "$DEST/"

echo "[transfer] done"
