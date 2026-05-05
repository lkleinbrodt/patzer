# Lichess scrape on a DigitalOcean droplet

Set your droplet once:

```bash
export PATZER_DROPLET='root@YOUR_DROPLET_IP'
```

From the **repo root** on your laptop:

```bash
./pipeline/droplet_scrape.sh sync
./pipeline/droplet_scrape.sh push-env    # needs ./.env with R2_* vars
./pipeline/droplet_scrape.sh setup       # once per fresh droplet
```

SSH in:

```bash
ssh "$PATZER_DROPLET"
```

Long run (survives disconnect): **tmux**, then scrape (use the **venv** Python — not bare `python3`, avoids PEP 668 / missing `chess`):

```bash
cd ~/patzer
tmux new -s lichess

.venv/bin/python pipeline/scrape_lichess.py \
  --output-dir data/lichess_games \
  --push-r2 \
  --min-month 2020-04 \
  --min-elo 1800 \
  2>&1 | tee scrape.log
```

Detach tmux: **Ctrl+b** then **d**. Reattach later:

```bash
ssh "$PATZER_DROPLET"
tmux attach -t lichess
```

After code changes locally, **`./pipeline/droplet_scrape.sh sync`** again before restarting the scraper.

### Fix: `ModuleNotFoundError: No module named 'chess'`

New Ubuntu blocks system-wide `pip install`. Re-run **`./pipeline/droplet_scrape.sh setup`** from your laptop (creates **`~/patzer/.venv`** and installs **`python-chess`**). Or on the droplet:

```bash
cd ~/patzer && python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install requests python-chess boto3 python-dotenv
```

Always run the scraper with **`~/patzer/.venv/bin/python`**, not plain **`python3`**.

The PyPI package is **`python-chess`**; **`pip install chess`** is the wrong package / may be blocked (PEP 668).
