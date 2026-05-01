#!/usr/bin/env python3
"""
Deploy / run Patzer through an external lichess-bot checkout.

  python bot/deploy_bot.py install-shim
  python bot/deploy_bot.py run v1
  python bot/deploy_bot.py run v2

Token lookup order (first non-empty wins):
  1. LICHESS_BOT_TOKEN env var
  2. PATZER_V<N>_TOKEN in .env  (e.g. PATZER_V2_TOKEN=lip_xxx)
  3. token: field in the bot's YAML config

Environment:
  PATZER_ROOT       Patzer repo root (default: parent of bot/)
  LICHESS_BOT_HOME  lichess-bot clone (default: ~/Projects/lichess-bot)
  LICHESS_BOT_TOKEN Lichess API token (overrides all other sources)

Config files (first match wins):
  bot/configs/<label>.local.yml
  bot/configs/<label>.yml

Run from Patzer repository root (recommended).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_lichess_bot_home() -> Path:
    return Path(os.environ.get("LICHESS_BOT_HOME", os.path.expanduser("~/Projects/lichess-bot")))


def normalize_label(arg: str) -> str:
    a = arg.strip().lower()
    if a.startswith("patzer_v"):
        return a
    if len(a) >= 2 and a[0] == "v" and a[1:].isdigit():
        return f"patzer_{a}"
    if a.isdigit():
        return f"patzer_v{a}"
    return f"patzer_{a}"


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Handles quotes and ignores comments."""
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        result[key] = val
    return result


def resolve_token(label: str, cfg: Path) -> str:
    """Find the Lichess token for this bot, checking all sources in priority order."""
    # 1. LICHESS_BOT_TOKEN already in environment
    token = os.environ.get("LICHESS_BOT_TOKEN", "").strip()
    if token:
        return token

    # 2. Per-bot key in .env: patzer_v2 → PATZER_V2_TOKEN
    env_key = label.upper() + "_TOKEN"
    dotenv = load_dotenv(repo_root() / ".env")
    token = dotenv.get(env_key, "").strip()
    if token:
        return token

    # 3. token: field in the YAML config
    import yaml  # provided by lichess-bot's venv

    with open(cfg) as f:
        cfg_data = yaml.safe_load(f)
    token = (cfg_data or {}).get("token", "").strip()
    if token:
        return token

    raise SystemExit(
        f"ERROR: No Lichess token found for {label}.\n"
        f"  Option A: add  {env_key}=lip_xxx  to .env\n"
        f"  Option B: set  LICHESS_BOT_TOKEN=lip_xxx  in your shell\n"
        f"  Option C: add  token: lip_xxx  to {cfg}"
    )


def resolve_config_path(label: str) -> Path:
    root = repo_root()
    local_p = root / "bot" / "configs" / f"{label}.local.yml"
    if local_p.is_file():
        return local_p
    p = root / "bot" / "configs" / f"{label}.yml"
    if p.is_file():
        return p
    raise FileNotFoundError(
        f"No config for '{label}': expected {local_p} or {p}. "
        f"Add {label}.yml under bot/configs/ or copy an existing one to {label}.local.yml (gitignored)."
    )


def pick_python(lichess_home: Path) -> str:
    venv_py = lichess_home / ".venv" / "bin" / "python"
    if venv_py.is_file():
        return str(venv_py)
    print("WARNING: no .venv found in lichess-bot dir — using current Python (may lack chess/berserk)")
    return sys.executable


def cmd_install_shim(lichess_home: Path, dry_run: bool) -> None:
    src = repo_root() / "bot" / "templates" / "homemade_shim.py"
    dst = lichess_home / "homemade.py"
    if not src.is_file():
        raise FileNotFoundError(src)
    if not lichess_home.is_dir():
        raise FileNotADirectoryError(
            f"Lichess-bot directory not found: {lichess_home}. Set LICHESS_BOT_HOME."
        )
    if dst.is_symlink() and dst.resolve() == src.resolve():
        print(f"Shim already linked: {dst} -> {src}")
        return
    print(f"Link shim: {dst} -> {src}")
    if dry_run:
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def cmd_upgrade(label: str, lichess_home: Path, dry_run: bool) -> None:
    cfg = resolve_config_path(label)
    lichess_py = lichess_home / "lichess-bot.py"
    if not lichess_py.is_file():
        raise FileNotFoundError(
            f"lichess-bot.py not found in {lichess_home}. Clone lichess-bot or set LICHESS_BOT_HOME."
        )

    patzer_root = os.environ.get("PATZER_ROOT", str(repo_root()))
    python = pick_python(lichess_home)
    argv = [python, str(lichess_py), "--config", str(cfg.resolve()), "-u"]
    print("Upgrading account for", label)
    print("exec:", " ".join(argv))
    if dry_run:
        return

    token = resolve_token(label, cfg)
    env = os.environ.copy()
    env["PATZER_ROOT"] = patzer_root
    env["LICHESS_BOT_TOKEN"] = token

    os.chdir(lichess_home)
    subprocess.run(argv, env=env, check=True)


def cmd_run(label: str, lichess_home: Path, dry_run: bool) -> None:
    cfg = resolve_config_path(label)
    lichess_py = lichess_home / "lichess-bot.py"
    if not lichess_py.is_file():
        raise FileNotFoundError(
            f"lichess-bot.py not found in {lichess_home}. Clone lichess-bot or set LICHESS_BOT_HOME."
        )

    patzer_root = os.environ.get("PATZER_ROOT", str(repo_root()))
    python = pick_python(lichess_home)
    argv = [python, str(lichess_py), "--config", str(cfg.resolve())]
    print("cwd:         ", lichess_home)
    print("exec:        ", " ".join(argv))
    print("PATZER_ROOT: ", patzer_root)
    if dry_run:
        return

    token = resolve_token(label, cfg)
    env = os.environ.copy()
    env["PATZER_ROOT"] = patzer_root
    env["LICHESS_BOT_TOKEN"] = token  # lichess-bot reads this to override the YAML token field

    os.chdir(lichess_home)
    subprocess.run(argv, env=env, check=True)


def main() -> None:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--lichess-bot-home",
        type=Path,
        default=None,
        help=f"Path to lichess-bot clone (default: {default_lichess_bot_home()})",
    )
    parent.add_argument("--dry-run", action="store_true", help="Print actions only")

    ap = argparse.ArgumentParser(description="Patzer + lichess-bot deploy helper")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "install-shim",
        parents=[parent],
        help="Symlink bot/templates/homemade_shim.py into lichess-bot/homemade.py (one-time setup)",
    )

    p_upgrade = sub.add_parser(
        "upgrade",
        parents=[parent],
        help="Upgrade a Lichess account to a bot account (irreversible)",
    )
    p_upgrade.add_argument("label", nargs="?", default="v1", help="Config name: v1, v2, …")

    p_run = sub.add_parser(
        "run",
        parents=[parent],
        help="Ensure shim is linked, then start lichess-bot with a Patzer config",
    )
    p_run.add_argument(
        "label",
        nargs="?",
        default="v1",
        help="Config name: v1, v2, patzer_v1, … (default: v1 → patzer_v1.yml)",
    )

    args = ap.parse_args()

    # Load .env into os.environ (shell env takes precedence)
    for k, v in load_dotenv(repo_root() / ".env").items():
        os.environ.setdefault(k, v)

    home = args.lichess_bot_home or default_lichess_bot_home()

    if args.command == "install-shim":
        cmd_install_shim(home, args.dry_run)
        return

    if args.command == "upgrade":
        label = normalize_label(args.label)
        cmd_upgrade(label, home, args.dry_run)
        return

    if args.command == "run":
        label = normalize_label(args.label)
        cmd_install_shim(home, args.dry_run)
        cmd_run(label, home, args.dry_run)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
