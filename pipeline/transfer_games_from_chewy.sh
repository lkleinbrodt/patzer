#!/usr/bin/env bash
# Back-compat wrapper.
# We used to sync `data/lichess_games/games_*.txt` from chewy.
# We now sync tokenized `data/prepared/` binaries instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/transfer_prepared_from_chewy.sh" "${1:-}"
