#!/usr/bin/env bash
# Sync Patzer to a DigitalOcean droplet and run Lichess scrape → R2.
#
#   export PATZER_DROPLET='root@your.droplet.ip'
#   ./pipeline/droplet_scrape.sh sync          # rsync code (no local data/)
#   ./pipeline/droplet_scrape.sh push-env      # copy .env from repo root (R2_* secrets)
#   ./pipeline/droplet_scrape.sh setup         # apt + pip deps on the droplet (once)
#   ./pipeline/droplet_scrape.sh run -- ...    # scrape_lichess.py args after --
#
# Example run (foreground):
#   ./pipeline/droplet_scrape.sh run -- --output-dir data/lichess_games --push-r2 \\
#     --min-month 2020-04 --min-elo 1800
#
# Example in tmux on the droplet (survives SSH disconnect):
#   ssh "$PATZER_DROPLET" 'tmux new -d -s lichess "cd ~/patzer && .venv/bin/python pipeline/scrape_lichess.py ... 2>&1 | tee scrape.log"'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DROPLET="${PATZER_DROPLET:-}"

die() { echo "$*" >&2; exit 1; }

need_droplet() {
  [ -n "$DROPLET" ] || die "Set PATZER_DROPLET, e.g. export PATZER_DROPLET=root@203.0.113.10"
}

RSYNC_EXCLUDES=(
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '.git/'
  --exclude 'data/'
  --exclude 'checkpoints/'
  --exclude 'eval/results.db'
  --exclude '*.pyc'
  --exclude '.DS_Store'
)

cmd="${1:-}"
shift || true

case "$cmd" in
  sync)
    need_droplet
    rsync -avz "${RSYNC_EXCLUDES[@]}" \
      "$REPO_ROOT/" "$DROPLET:~/patzer/"
    echo "[droplet] synced → $DROPLET:~/patzer/"
    ;;
  push-env)
    need_droplet
    envfile="$REPO_ROOT/.env"
    [ -f "$envfile" ] || die "Missing $envfile (create it with R2_* vars)"
    ssh "$DROPLET" mkdir -p ~/patzer
    scp "$envfile" "$DROPLET:~/patzer/.env"
    echo "[droplet] copied .env → $DROPLET:~/patzer/.env"
    ;;
  setup)
    need_droplet
    ssh "$DROPLET" bash -s <<'REMOTE'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq wget zstd ca-certificates python3 python3-pip python3-venv python3-full git
mkdir -p ~/patzer
cd ~/patzer
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q requests python-chess boto3 python-dotenv
echo "[droplet] venv at ~/patzer/.venv + scrape pip deps OK"
REMOTE
    ;;
  run)
    need_droplet
    [ "${1:-}" = "--" ] && shift
    ssh -t "$DROPLET" "cd ~/patzer && .venv/bin/python pipeline/scrape_lichess.py $*"
    ;;
  "")
    die "usage: PATZER_DROPLET=root@host $0 sync|push-env|setup|run -- [scrape_lichess.py args]"
    ;;
  *)
    die "unknown command: $cmd"
    ;;
esac
