#!/usr/bin/env python3
"""
Rotate through Patzer lichess-bot configs: run each for a fixed wall time, then SIGINT once.

Requires lichess-bot configs with quit_after_all_games_finish: true (default in bot/configs).

Run from repo root:
  python bot/cycle_bots.py
  python bot/cycle_bots.py v1 v3    # subset

With no positional args, discovers every patzer_*.yml / patzer_*.local.yml under bot/configs/
(same labels as deploy_bot). Default dwell is 3600s.

Notifications (same as launch.py / Vast): set NTFY_TOPIC in the environment or .env to get
alerts on unexpected bot exits, cycler errors, or shutdowns that exceed --max-wait-after-sigint.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from deploy_bot import (
    cmd_install_shim,
    default_lichess_bot_home,
    load_dotenv,
    normalize_label,
    prepare_run,
    repo_root,
)

# So main() can forward SIGINT to the active lichess-bot on Ctrl-C
_child_proc: subprocess.Popen | None = None

# “Gentle” exit codes after we have sent SIGINT (don’t ntfy on these)
_GRACEFUL_EXIT_CODES = {
    0,
    -signal.SIGINT,
    128 + signal.SIGINT,  # 130 when WIFEXITED
}


def _ntfy_send(title: str, message: str, priority: str = "default") -> None:
    topic = (os.environ.get("NTFY_TOPIC") or "").strip()
    if not topic:
        return
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url,
        data=message.encode("utf-8"),
        method="POST",
    )
    req.add_header("Title", title)
    req.add_header("Priority", priority)
    req.add_header("Tags", "chess,robot")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


def discover_patzer_bot_labels(config_dir: Path) -> list[str]:
    """
    Labels that have a matching YAML under config_dir (patzer_*.yml or patzer_*.local.yml).

    Matches deploy_bot naming; sorted with patzer_v1 … patzer_v9 … patzer_v10 first, then others.
    """
    found: set[str] = set()
    if not config_dir.is_dir():
        return []
    for path in config_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        m = re.fullmatch(r"patzer_(.+)\.local\.yml", name)
        if m:
            found.add(f"patzer_{m.group(1)}")
            continue
        m = re.fullmatch(r"patzer_(.+)\.yml", name)
        if m:
            found.add(f"patzer_{m.group(1)}")

    def sort_key(lab: str) -> tuple:
        m = re.fullmatch(r"patzer_v(\d+)$", lab)
        if m:
            return (0, int(m.group(1)))
        return (1, lab)

    return sorted(found, key=sort_key)


def _exit_code_ok_after_sigint(code: int | None) -> bool:
    if code is None:
        return False
    if code in _GRACEFUL_EXIT_CODES:
        return True
    # Negative = killed by signal on Unix (e.g. -2 for SIGINT)
    if code < 0 and code == -signal.SIGINT:
        return True
    return False


def _run_one_bot(
    label: str,
    lichess_home: Path,
    dwell_sec: float,
    max_wait_after_sigint: float,
) -> None:
    global _child_proc
    argv, env, cwd = prepare_run(label, lichess_home)
    print(f"--- cycle: starting {label} (dwell {dwell_sec:.0f}s) ---", flush=True)
    proc = subprocess.Popen(
        argv,
        env=env,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
    )
    _child_proc = proc
    sigint_sent = False
    deadline = time.monotonic() + dwell_sec
    try:
        while True:
            code = proc.poll()
            if code is not None:
                _ntfy_send(
                    "Patzer bot cycle: bot exited early",
                    f"{label} stopped before rotate time (exit {code}). "
                    "Check logs / Lichess; cycler continues to next bot.",
                    priority="high",
                )
                print(f"--- cycle: {label} exited early with code {code} ---", flush=True)
                return

            now = time.monotonic()
            if now >= deadline:
                break

            time.sleep(min(60.0, deadline - now))

        sigint_sent = True
        print(
            f"--- cycle: dwell done; sending SIGINT to {label} (wait for games to finish) ---",
            flush=True,
        )
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

        join_deadline = time.monotonic() + max_wait_after_sigint
        while proc.poll() is None:
            if time.monotonic() >= join_deadline:
                _ntfy_send(
                    "Patzer bot cycle: shutdown still in progress",
                    f"{label} is still running {max_wait_after_sigint:.0f}s after SIGINT — "
                    "likely long game(s). Do not kill the process. "
                    "The cycler will keep waiting; check the machine and Lichess.",
                    priority="urgent",
                )
                join_deadline = time.monotonic() + 3600.0
            time.sleep(min(30.0, 60.0))

        code = proc.returncode
        if not _exit_code_ok_after_sigint(code):
            _ntfy_send(
                "Patzer bot cycle: odd exit after SIGINT",
                f"{label} exited with code {code} after graceful interrupt. "
                "Worth a quick look at lichess-bot logs.",
                priority="default",
            )
        print(f"--- cycle: {label} finished (exit {code}) ---", flush=True)
    except Exception:
        _ntfy_send(
            "Patzer bot cycle: internal error",
            f"While running {label}:\n{traceback.format_exc()}",
            priority="urgent",
        )
        raise
    finally:
        if proc.poll() is None and not sigint_sent:
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        if proc.poll() is None:
            print(
                f"--- cycle: waiting for {label} (pid {proc.pid}) to exit ---",
                flush=True,
            )
            proc.wait()
        _child_proc = None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run Patzer lichess bots in rotation with graceful SIGINT between slots."
    )
    ap.add_argument(
        "labels",
        nargs="*",
        metavar="LABEL",
        help="Bot labels (v1, patzer_v2, …). Default: auto-discover bot/configs/patzer_*.yml "
        "and patzer_*.local.yml",
    )
    ap.add_argument(
        "--dwell",
        type=float,
        default=3600.0,
        metavar="SEC",
        help="Seconds to run each bot before sending SIGINT (default: 3600 = 1h)",
    )
    ap.add_argument(
        "--max-wait-after-sigint",
        type=float,
        default=6 * 3600.0,
        metavar="SEC",
        help="If the process is still alive this long after SIGINT, send an urgent ntfy "
        "(default: 6h). Waits indefinitely after that with hourly reminders.",
    )
    ap.add_argument(
        "--lichess-bot-home",
        type=Path,
        default=None,
        help=f"Lichess-bot clone (default: {default_lichess_bot_home()})",
    )
    ap.add_argument(
        "--no-install-shim",
        action="store_true",
        help="Skip symlinking homemade.py (use if you already ran install-shim)",
    )
    args = ap.parse_args()

    for k, v in load_dotenv(repo_root() / ".env").items():
        os.environ.setdefault(k, v)

    home = args.lichess_bot_home or default_lichess_bot_home()
    if not args.no_install_shim:
        cmd_install_shim(home, dry_run=False)

    if args.labels:
        labels = [normalize_label(x) for x in args.labels]
    else:
        labels = discover_patzer_bot_labels(repo_root() / "bot" / "configs")
        if not labels:
            print(
                "ERROR: No bot configs found (expected bot/configs/patzer_*.yml or patzer_*.local.yml).",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.dwell <= 0:
        print("ERROR: --dwell must be positive", file=sys.stderr)
        sys.exit(2)
    if args.max_wait_after_sigint <= 0:
        print("ERROR: --max-wait-after-sigint must be positive", file=sys.stderr)
        sys.exit(2)

    if (os.environ.get("NTFY_TOPIC") or "").strip():
        print("Notifications: NTFY_TOPIC is set", flush=True)
    else:
        print("Notifications: NTFY_TOPIC unset (no push alerts)", flush=True)
    print(f"Rotation order: {labels}  dwell={args.dwell}s", flush=True)

    try:
        while True:
            for label in labels:
                _run_one_bot(
                    label,
                    home,
                    dwell_sec=args.dwell,
                    max_wait_after_sigint=args.max_wait_after_sigint,
                )
    except KeyboardInterrupt:
        print("--- cycle: KeyboardInterrupt, exiting ---", flush=True)
        c = _child_proc
        if c is not None and c.poll() is None:
            try:
                c.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        _ntfy_send(
            "Patzer bot cycle: operator stopped",
            "cycle_bots.py received Ctrl-C. If a lichess-bot was still up, a SIGINT was sent; "
            "press Ctrl-C again in the child or use force_quit per lichess-bot docs if needed.",
            priority="default",
        )
        raise


if __name__ == "__main__":
    main()
