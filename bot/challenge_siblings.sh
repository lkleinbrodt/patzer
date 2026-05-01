#!/usr/bin/env bash
#
# Bash 3 compatible (macOS default). Edit the CONFIG section to choose bots,
# token env var, and time controls.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Load tokens/config from repo .env if present (without requiring it).
if [[ -f "$ROOT_DIR/.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
fi

### CONFIG (edit me)
OPPONENT_USERNAME="patzer_v1"

# Token must belong to the challenger account (default: patzer_v3).
# stored in .env as PATZER_V3_TOKEN
CHALLENGER_TOKEN="${PATZER_V3_TOKEN}"

# Rated games? ("true" or "false")
RATED="true"

# Seconds to sleep between challenges (avoid rate limits / spam)
SLEEP_S="2"

# Presets are parallel arrays: NAME / LIMIT_SECONDS / INCREMENT_SECONDS / NUM_GAMES
TC_NAMES=("bullet_1+0" "bullet_1+1" "blitz_3+0" "blitz_3+2" "blitz_5+0")
TC_LIMITS=(60          60          180        180        300)
TC_INCS=(0             1           0          2          0)
TC_COUNTS=(1           1           1          1          1)
### /CONFIG

if [[ -z "${CHALLENGER_TOKEN}" ]]; then
  echo "Missing challenger token." >&2
  echo "Set one of: LICHESS_BOT_TOKEN_PATZER_V3, LICHESS_BOT_TOKEN_V3, or LICHESS_BOT_TOKEN (in .env is fine)." >&2
  exit 1
fi

if [[ ${#TC_NAMES[@]} -ne ${#TC_LIMITS[@]} || ${#TC_NAMES[@]} -ne ${#TC_INCS[@]} || ${#TC_NAMES[@]} -ne ${#TC_COUNTS[@]} ]]; then
  echo "Time-control arrays are mismatched lengths. Fix TC_* arrays in CONFIG." >&2
  exit 2
fi

echo "Opponent: $OPPONENT_USERNAME"
echo "Rated:    $RATED"
echo "Presets:  ${TC_NAMES[*]}"

i=0
while [[ $i -lt ${#TC_NAMES[@]} ]]; do
  name="${TC_NAMES[$i]}"
  limit="${TC_LIMITS[$i]}"
  inc="${TC_INCS[$i]}"
  n="${TC_COUNTS[$i]}"

  echo "Sending $n challenges: $name (${limit}+${inc})"
  j=0
  while [[ $j -lt $n ]]; do
    curl -sS -X POST "https://lichess.org/api/challenge/$OPPONENT_USERNAME" \
      -H "Authorization: Bearer $CHALLENGER_TOKEN" \
      -d rated="$RATED" \
      -d variant=standard \
      -d clock.limit="$limit" \
      -d clock.increment="$inc" \
      -d color=random >/dev/null
    sleep "$SLEEP_S"
    j=$((j + 1))
  done

  i=$((i + 1))
done

