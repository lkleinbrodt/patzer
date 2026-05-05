#!/usr/bin/env bash
# Compress games_YYYY-MM.txt → games_YYYY-MM.txt.gz when .gz is absent,
# then verify line count vs games_YYYY-MM.stats.json (kept_games). If it
# matches, remove the plain .txt.
#
# Usage:
#   ./pipeline/compress_games_txt.sh              # default: data/lichess_games
#   ./pipeline/compress_games_txt.sh /path/to/dir
#
# Requires: gzip, jq, wc

set -euo pipefail

DIR="${1:-data/lichess_games}"

if [[ ! -d "$DIR" ]]; then
  echo "error: not a directory: $DIR" >&2
  exit 1
fi

shopt -s nullglob

failures=0
for txt in "$DIR"/games_*.txt; do
  [[ -f "$txt" ]] || continue

  base="${txt%.txt}"           # .../games_2020-04
  gz="${base}.txt.gz"
  stats="${base}.stats.json"

  month="$(basename "$base")"  # games_2020-04

  if [[ ! -f "$stats" ]]; then
    echo "skip $month: missing stats $(basename "$stats")" >&2
    failures=1
    continue
  fi

  if [[ ! -f "$gz" ]]; then
    echo "compress $month → $(basename "$gz")"
    gzip -6 -k "$txt"
  fi

  if [[ ! -f "$gz" ]]; then
    echo "error $month: expected $(basename "$gz") after gzip" >&2
    failures=1
    continue
  fi

  kept="$(jq -r '.kept_games // empty' "$stats" 2>/dev/null || true)"
  if [[ -z "$kept" || "$kept" == "null" ]]; then
    echo "error $month: could not read .kept_games from $(basename "$stats")" >&2
    failures=1
    continue
  fi

  lines="$(gzip -dc "$gz" | wc -l | tr -d ' ')"
  if [[ "$lines" != "$kept" ]]; then
    echo "error $month: line count mismatch (gzip lines=$lines, stats kept_games=$kept)" >&2
    failures=1
    continue
  fi

  echo "ok $month: $lines lines == kept_games; removing $(basename "$txt")"
  rm -f "$txt"
done

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
