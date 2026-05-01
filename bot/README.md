# Lichess bot (Patzer)

## One-time setup

```bash
# 1. Clone lichess-bot and install its deps
git clone https://github.com/lichess-bot-devs/lichess-bot ~/Projects/lichess-bot
cd ~/Projects/lichess-bot && python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Link the Patzer engine shim (run from Patzer repo root)
python bot/deploy_bot.py install-shim
```

## Tokens

Add a `.env` file at the repo root (gitignored):

```
PATZER_V1_TOKEN=lip_xxx
PATZER_V2_TOKEN=lip_yyy

# Optional — only needed if lichess-bot isn't at ~/Projects/lichess-bot
LICHESS_BOT_HOME=/path/to/lichess-bot
```

Get tokens from https://lichess.org/account/oauth/token — the account must already be upgraded to a bot account.

## Deploy a bot

```bash
python bot/deploy_bot.py run v2
```

That's it. It picks up the token from `.env`, links the shim if needed, and starts lichess-bot with `bot/configs/patzer_v2.yml`.

Run multiple bots simultaneously in separate terminals:

```bash
python bot/deploy_bot.py run v1   # terminal 1
python bot/deploy_bot.py run v2   # terminal 2
```

## Add a new version

```bash
cp bot/configs/patzer_v2.yml bot/configs/patzer_v3.yml
# edit patzer_checkpoint, greeting, etc.
echo "PATZER_V3_TOKEN=lip_zzz" >> .env
python bot/deploy_bot.py run v3
```

## Config reference

`engine.homemade_options` in each YAML:

| Key | Default | Description |
|-----|---------|-------------|
| `patzer_checkpoint` | — (required) | path relative to Patzer repo root, e.g. `checkpoints/patzer_v2/weights_best.pt` |
| `device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `temperature` | `0` | 0 = greedy argmax over legal moves |
| `conditioning` | `match_color` | passed to `eval.engine.Patzer` |

Token lookup order (first non-empty wins): `LICHESS_BOT_TOKEN` shell var → `PATZER_V<N>_TOKEN` in `.env` → `token:` in the YAML.
