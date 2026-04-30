#!/usr/bin/env bash
# Sync processed Lichess month lines from chewy → this repo.
# Only games_*.txt (one game per line). No raw dumps: *.zst / *.pgn.zst are excluded.
# Exclude nested data/lichess_games/lichess_games/ so a bad remote layout cannot recurse.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/lichess_games

rsync -avP --partial --append-verify --checksum \
  --exclude 'lichess_games/' \
  --include 'games_*.txt' \
  --exclude '*' \
  "chewy:/home/landon/Projects/patzer/data/lichess_games/" \
  "data/lichess_games/"
