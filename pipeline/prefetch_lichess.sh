#!/usr/bin/env bash
# Prefetch Lichess monthly dumps in parallel with scraping.
#
# Keeps exactly ONE month ahead downloaded (resumable) based on:
#   - output_dir/progress.json (completed months)
#   - any existing lichess_db_standard_rated_YYYY-MM.pgn.zst files in output_dir
#
# This plays nicely with pipeline/scrape_lichess.py, which downloads+processes
# sequentially. Run this in a separate tmux pane while scrape_lichess.py runs.
#
# Usage:
#   ./pipeline/prefetch_lichess.sh data/lichess_games
#
# Options via env:
#   SLEEP_SECS=60        # poll interval
#   NICE=10              # CPU nice
#   IONICE_CLASS=2       # 2=best-effort, 3=idle
#   IONICE_LEVEL=7       # 0..7 (7 = lowest priority)
#
# Requirements: bash, wget, python3, jq (optional but recommended)

set -euo pipefail

OUTPUT_DIR="${1:-data/lichess_games}"
SLEEP_SECS="${SLEEP_SECS:-60}"
NICE="${NICE:-10}"
IONICE_CLASS="${IONICE_CLASS:-2}"
IONICE_LEVEL="${IONICE_LEVEL:-7}"

BASE_URL="https://database.lichess.org/standard"
PREFIX="lichess_db_standard_rated_"
SUFFIX=".pgn.zst"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "error: not a directory: $OUTPUT_DIR" >&2
  exit 1
fi

# Prevent multiple prefetchers in same directory.
LOCKFILE="$OUTPUT_DIR/.prefetch.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "error: another prefetch_lichess.sh is already running for $OUTPUT_DIR" >&2
  exit 2
fi

month_add() {
  # Add N months to YYYY-MM using python (portable on Linux).
  local m="$1"
  local add="$2"
  python3 - "$m" "$add" <<'PY'
import sys
from datetime import date

s = sys.argv[1]
add = int(sys.argv[2])
y, mo = map(int, s.split("-"))
idx = (y * 12 + (mo - 1)) + add
y2, m2 = divmod(idx, 12)
print(f"{y2:04d}-{m2+1:02d}")
PY
}

latest_completed_month() {
  local progress="$OUTPUT_DIR/progress.json"
  if [[ ! -f "$progress" ]]; then
    echo ""
    return
  fi

  if command -v jq >/dev/null 2>&1; then
    jq -r 'map(select(type=="string")) | sort | last // empty' "$progress" 2>/dev/null || true
  else
    # Fallback: crude parse. (Install jq if you can.)
    python3 - "$progress" <<'PY'
import json, sys
p = sys.argv[1]
try:
  arr = json.load(open(p))
except Exception:
  print("")
  raise SystemExit(0)
arr = [x for x in arr if isinstance(x, str)]
arr.sort()
print(arr[-1] if arr else "")
PY
  fi
}

latest_downloaded_month() {
  # Highest month among any .pgn.zst files present (downloaded/in-flight/prefetched).
  local best=""
  shopt -s nullglob
  for p in "$OUTPUT_DIR"/${PREFIX}????-??${SUFFIX}; do
    local b
    b="$(basename "$p")"
    local m="${b#$PREFIX}"
    m="${m%$SUFFIX}"
    if [[ "$m" =~ ^[0-9]{4}-[0-9]{2}$ ]]; then
      if [[ -z "$best" || "$m" > "$best" ]]; then
        best="$m"
      fi
    fi
  done
  echo "$best"
}

prefetch_one() {
  local month="$1"
  local out="$OUTPUT_DIR/${PREFIX}${month}${SUFFIX}"
  local url="$BASE_URL/${PREFIX}${month}${SUFFIX}"

  if [[ -f "$out" ]]; then
    echo "[prefetch] already have $month ($(basename "$out"))"
    return 0
  fi

  echo "[prefetch] downloading $month → $(basename "$out")"
  # Deprioritize so scraper processing stays fast.
  nice -n "$NICE" ionice -c"$IONICE_CLASS" -n"$IONICE_LEVEL" \
    wget --continue --progress=bar:force --tries=5 --waitretry=30 --timeout=60 \
      --output-document "$out" \
      "$url"
}

echo "[prefetch] output_dir=$OUTPUT_DIR (poll every ${SLEEP_SECS}s)"
echo "[prefetch] nice=$NICE ionice=${IONICE_CLASS}:${IONICE_LEVEL}"

while true; do
  c="$(latest_completed_month)"
  d="$(latest_downloaded_month)"

  # If scraper has completed through month C, it's probably processing C+1 (and
  # has that .zst present unless it already deleted it). So we aim to download C+2.
  # But if there's already a higher downloaded month present, use that as the base.
  base="$c"
  if [[ -z "$base" ]]; then
    base="$d"
  elif [[ -n "$d" && "$d" > "$base" ]]; then
    base="$d"
  fi

  if [[ -z "$base" ]]; then
    echo "[prefetch] waiting for progress.json or a downloaded .zst to appear…"
    sleep "$SLEEP_SECS"
    continue
  fi

  # Keep one month ahead:
  # - If base comes from completed month C: download C+2
  # - If base comes from existing .zst (likely current processing): download (that)+1
  target=""
  if [[ -n "$d" && "$base" == "$d" ]]; then
    target="$(month_add "$base" 1)"
  else
    target="$(month_add "$base" 2)"
  fi

  prefetch_one "$target" || true
  sleep "$SLEEP_SECS"
done

