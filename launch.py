"""
Launch Vast.ai GPU instances for patzer training.

Before any rent or --instance SSH, launch.py verifies `patzer/config/<name>.py` exists,
compiles, and executes locally (exit 2 on failure). --list / --status skip this check.

Default search includes ``cuda_vers>=12.4`` so hosts report driver ABI compatible with our
CUDA 12.4 PyTorch image; override with ``--min-cuda-vers 0`` to disable.

Default search includes ``compute_cap<=900`` so GPUs match PyTorch 2.5+cu12.4 prebuilts (through
sm_90) and Blackwell (sm_120, ~1200) is excluded. Use ``--max-compute-cap 0`` after upgrading
the training image / PyTorch for sm_120 support.

Bandwidth cost filters are **off** by default (``inet_*`` unrestricted). Use
``--max-inet-up-cost 0 --max-inet-down-cost 0`` for upload/download-free hosts only.

  python launch.py train                      # rent cheapest offer, confirm prompt
  python launch.py train --search-only        # print offers and exit
  python launch.py train --list               # show your running instances
  python launch.py train --instance 12345678  # train on an existing instance (in-place update; keeps data/)
  python launch.py train --instance 12345678 --full-reset  # wipe /workspace/patzer and fresh clone
  python launch.py train --config train_patzer_v1
  python launch.py train --resume             # pull R2 checkpoint and pass --init_from=resume

Legacy (still works): ``python launch.py --config train_patzer_v1`` inserts the ``train`` subcommand.

Lichess scraping runs on your own host (e.g. DigitalOcean): see ``pipeline/droplet_scrape.sh`` and CLAUDE.md.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

GITHUB_REPO = "https://github.com/lkleinbrodt/patzer.git"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
# Vast `compute_cap` is CUDA capability × 100 (e.g. 8.9 → 890, 9.0 → 900, 12.0 → 1200).
# Prebuilt PyTorch 2.5.x + CUDA 12.4 wheels ship kernels through sm_90 only → cap 900.
# Blackwell (sm_120) is ~1200 and triggers runtime warnings/errors unless PyTorch is upgraded.
DEFAULT_MAX_COMPUTE_CAP = 900
WORKSPACE = "/workspace/patzer"
VAST_API_KEY_PATH = Path.home() / ".config" / "vastai" / "vast_api_key"

# Curated list of GPU models worth renting for training.
# Matched as case-insensitive substrings against Vast's `gpu_name` field.
# Excludes mid-range cards (e.g. 4070) that are priced similarly to 3090 but with less VRAM.
DESIRED_GPUS = [
    "RTX 3060",
    "RTX 3070",
    "RTX 3080",
    "RTX 3090",
    "RTX 4080",
    "RTX 4090",
    "A4000",
    "A5000",
    "A6000",
    "L4",
    "L40",   # matches L40 and L40S
    "A100",
    "H100",
]


def _vast_api_key() -> str | None:
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key
    if VAST_API_KEY_PATH.exists():
        return VAST_API_KEY_PATH.read_text().strip()
    return None


def _wandb_api_key() -> str | None:
    return os.environ.get("WANDB_API_KEY") or os.environ.get("wandb_api_key")


def vast(*args, raw=True):
    cmd = ["vastai"] + list(args) + (["--raw"] if raw else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return json.loads(r.stdout) if raw else r.stdout.strip()


def _ntfy_topic() -> str | None:
    return os.environ.get("NTFY_TOPIC")


def r2_env_flags() -> str:
    keys = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
            "R2_ENDPOINT_URL", "R2_ACCOUNT_ID"]
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        print(f"WARNING: R2 env vars not set: {missing}")
    pairs = {k: os.environ[k] for k in keys if os.environ.get(k)}
    api_key = _vast_api_key()
    if api_key:
        pairs["VAST_API_KEY"] = api_key
    wb = _wandb_api_key()
    if wb:
        pairs["WANDB_API_KEY"] = wb
    ntfy = _ntfy_topic()
    if ntfy:
        pairs["NTFY_TOPIC"] = ntfy
    return " ".join(f"-e {k}={v}" for k, v in pairs.items())


def r2_export_lines() -> str:
    """Shell export lines for R2 + Vast vars, used in scripts running on existing instances."""
    keys = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET",
            "R2_ENDPOINT_URL", "R2_ACCOUNT_ID"]
    lines = [f"export {k}='{os.environ[k]}'" for k in keys if os.environ.get(k)]
    api_key = _vast_api_key()
    if api_key:
        lines.append(f"export VAST_API_KEY='{api_key}'")
    wb = _wandb_api_key()
    if wb:
        lines.append(f"export WANDB_API_KEY='{wb}'")
    ntfy = _ntfy_topic()
    if ntfy:
        lines.append(f"export NTFY_TOPIC='{ntfy}'")
    return "\n".join(lines)


def build_script(
    config: str,
    resume: bool,
    wipe: bool = False,
    terminate_on_error: bool = False,
    wandb_run_id: str | None = None,
) -> str:
    cfg = _read_train_config_vars(config)
    out_dir = str(cfg.get("out_dir") or "").strip()
    # For resume we only need the full optimizer checkpoint (`ckpt.pt`).
    # We optionally pull `weights_best.pt` + `metrics.jsonl` for convenience/debugging.
    pull_ckpt = ""
    if resume and out_dir:
        pull_ckpt = "\n".join([
            f"python -c \"import r2; r2.pull_file('{out_dir}/ckpt.pt')\" || true",
            f"python -c \"import r2; r2.pull_file('{out_dir}/weights_best.pt')\" || true",
            f"python -c \"import r2; r2.pull_file('{out_dir}/metrics.jsonl')\" || true",
        ])
    elif resume:
        # Fallback if out_dir couldn't be parsed (should be rare): pull all checkpoints.
        pull_ckpt = "python -c \"import r2; r2.pull_dir('checkpoints', 'checkpoints')\" || true"
    resume_flag = "--init_from=resume" if resume else ""
    wandb_resume_flag = f"--wandb_run_id={wandb_run_id}" if (wandb_run_id and resume) else ""

    # Vast.ai injects $CONTAINER_ID inside the container — that's the instance ID.
    # We install vastai so the script can self-destruct when done.
    kill_on_error = "1" if terminate_on_error else "0"
    notify_fn = """\
_notify() {
    local title="$1" msg="$2" priority="${3:-default}"
    [ -n "${NTFY_TOPIC:-}" ] || return 0
    curl -s -X POST \
        -H "Title: $title" \
        -H "Priority: $priority" \
        -H "Tags: chess,robot" \
        -d "$msg" \
        "https://ntfy.sh/${NTFY_TOPIC}" || true
}"""
    trap = f"""\
_on_exit() {{
    EXIT_CODE=$?
    echo "=== Training exited (code $EXIT_CODE) ==="
    cd {WORKSPACE}/patzer && python -c "
import sys; sys.path.insert(0, '.')
import r2
r2.push_file('/workspace/train.log', 'logs/train_${{CONTAINER_ID:-unknown}}.log')
print('[r2] log pushed')
" || echo "[warn] log push to R2 failed"
    if [ $EXIT_CODE -eq 0 ]; then
        _notify "Patzer: training complete" "Instance ${{CONTAINER_ID:-unknown}} is being destroyed. Check W&B for results." "default"
    elif [ {kill_on_error} -eq 1 ]; then
        _notify "Patzer: training FAILED — instance destroyed" "Exit code $EXIT_CODE. Instance ${{CONTAINER_ID:-unknown}} destroyed." "high"
    else
        _notify "Patzer: training FAILED — instance ALIVE" "Exit code $EXIT_CODE. Instance ${{CONTAINER_ID:-unknown}} still running and costing money!" "urgent"
    fi
    if [ $EXIT_CODE -eq 0 ] || [ {kill_on_error} -eq 1 ]; then
        if [ -z "${{CONTAINER_ID:-}}" ]; then
            echo "[warn] CONTAINER_ID not set; cannot auto-destroy instance"
            return 0
        fi
        echo "Destroying instance ${{CONTAINER_ID}}..."
        # vastai 0.3.x prompts for confirmation; pipe 'y' so this works non-interactively.
        printf "y\n" | vastai --api-key "$VAST_API_KEY" destroy instance "$CONTAINER_ID" || true
        # Best-effort verification: after destroy, show should fail.
        sleep 2
        if vastai --api-key "$VAST_API_KEY" show instance "$CONTAINER_ID" >/dev/null 2>&1; then
            echo "[warn] destroy command returned but instance still appears to exist: $CONTAINER_ID"
        else
            echo "[vast] instance destroyed: $CONTAINER_ID"
        fi
    else
        echo "Training failed — instance kept alive for debugging (id=${{CONTAINER_ID}})"
        echo "SSH in to inspect, then: vastai destroy instance ${{CONTAINER_ID}}"
    fi
}}
trap _on_exit EXIT"""

    return "\n".join(filter(None, [
        "#!/bin/bash",
        "set -eo pipefail",
        # On new instances R2 vars come from Docker env; on existing instances
        # we write them explicitly so they're available to the training process.
        r2_export_lines(),
        "env | grep -E '^R2_' >> /etc/environment",
        # Existing instances often have large `data/` trees already downloaded.
        # Default behavior is to KEEP them and update the repo in-place.
        # Use --full-reset (wipe=True) to drop the whole workspace and reclone.
        f"rm -rf {WORKSPACE}" if wipe else "",
        (
            # Fresh clone if missing OR after wipe. Otherwise in-place update.
            # NOTE: `git clean` deletes untracked AND gitignored files, so we must explicitly
            # keep `patzer/data/` by default (large memmap binaries live there).
            f"if [ -d '{WORKSPACE}/.git' ]; then "
            f"cd {WORKSPACE} && git fetch --all --prune && git reset --hard origin/main && "
            f"git clean -fd -e patzer/data; "
            f"else git clone {GITHUB_REPO} {WORKSPACE}; fi"
        ),
        f"cd {WORKSPACE}",
        "pip install -q -r requirements.txt",
        "pip install -q vastai",
        notify_fn,
        trap,
        "cd patzer",
        # Before R2 data pull / train: fail fast on driver–container CUDA mismatch.
        "python -u -c 'import sys, torch; ok=torch.cuda.is_available(); "
        'print("torch.cuda.is_available():", ok); '
        'print(torch.cuda.get_device_name(0) if ok else "(none)"); '
        "sys.exit(0 if ok else 3)'",
        pull_ckpt,
        (
            f"if [ ! -f '{out_dir}/ckpt.pt' ]; then "
            f"echo \"[fatal] --resume requested but missing {out_dir}/ckpt.pt (R2 pull may have failed).\"; "
            f"exit 2; "
            f"fi"
            if resume and out_dir
            else ""
        ),
        f'_notify "Patzer: training started" "Config: {config}  Instance: ${{CONTAINER_ID:-unknown}}" "default"',
        f"python -u train.py config/{config}.py {resume_flag} {wandb_resume_flag} 2>&1 | tee /workspace/train.log".strip(),
    ]))


def print_offers(offers):
    def fmt_money(x: float | None, width: int = 7) -> str:
        if x is None:
            return " " * (width - 1) + "?"
        return f"{x:>{width}.3f}"

    def fmt_cost(x: float | None, width: int = 8) -> str:
        if x is None:
            return " " * (width - 1) + "?"
        return f"{x:>{width}.4f}"

    print(
        f"\n  {'#':>3}  {'GPU':<28} {'CUDA':>5} {'VRAM':>6} {'Base$/hr':>8} {'All-in$/hr':>9} "
        f"{'Up$/GB':>7} {'Dn$/GB':>7} {'Reliability':>11}  {'ID':>10}"
    )
    print("  " + "-" * 112)
    for i, o in enumerate(offers):
        vram_gb = int(o["gpu_ram"]) // 1024
        base = o.get("dph_total")
        up = o.get("inet_up_cost")
        dn = o.get("inet_down_cost")
        all_in = o.get("_all_in_dph")
        rel = o.get("reliability2")
        cuda_v = o.get("cuda_max_good")
        if cuda_v is None:
            cuda_v = o.get("cuda_vers")
        try:
            cuda_s = f"{float(cuda_v):.1f}" if cuda_v is not None else "?"
        except (TypeError, ValueError):
            cuda_s = "?"
        print(
            f"  {i+1:>3}  {o['gpu_name']:<28} {cuda_s:>5} {vram_gb:>5}G "
            f"{fmt_money(base, 8)} {fmt_money(all_in, 9)} "
            f"{fmt_cost(up, 7)} {fmt_cost(dn, 7)} "
            f"{(rel if rel is not None else 0):>11.3f}  {o['id']:>10}"
        )


def _offer_all_in_dph(
    offer: dict,
    *,
    up_gb_per_hr: float,
    down_gb_per_hr: float,
) -> float:
    """Estimate $/hr including expected network transfer costs.

    Vast exposes inet_up_cost/inet_down_cost in $/GB.
    """
    base = float(offer.get("dph_total") or 0.0)
    up_cost = float(offer.get("inet_up_cost") or 0.0)
    dn_cost = float(offer.get("inet_down_cost") or 0.0)
    return base + up_cost * up_gb_per_hr + dn_cost * down_gb_per_hr


_CONFIG_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_train_config(config_name: str) -> Path:
    """
    Ensure `patzer/config/{config_name}.py` exists, is under the repo config dir,
    compiles, and executes (same as `train.py` would `exec` the file).
    Call before any Vast SSH/rent path so a typo fails fast locally.
    """
    if not _CONFIG_NAME_RE.match(config_name):
        print(
            f"Invalid --config {config_name!r}: expected a basename like train_patzer_v5 "
            "(letters, digits, underscore; no path or .py suffix).",
            file=sys.stderr,
        )
        sys.exit(2)
    repo = Path(__file__).resolve().parent
    cfg_dir = (repo / "patzer" / "config").resolve()
    cfg_path = (cfg_dir / f"{config_name}.py").resolve()
    try:
        cfg_path.relative_to(cfg_dir)
    except ValueError:
        print(f"Refusing config path outside {cfg_dir}: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    if not cfg_path.is_file():
        print(
            f"Config not found: {cfg_path}\n"
            f"Expected patzer/config/{config_name}.py under the repo root.",
            file=sys.stderr,
        )
        sys.exit(2)
    source = cfg_path.read_text(encoding="utf-8")
    try:
        code = compile(source, str(cfg_path), "exec")
    except SyntaxError as e:
        print(f"Syntax error in config {cfg_path}:\n{e}", file=sys.stderr)
        sys.exit(2)
    ns: dict[str, object] = {
        "__builtins__": __builtins__,
        "__name__": "__patzer_launch_config_check__",
        "__file__": str(cfg_path),
    }
    try:
        exec(code, ns, ns)
    except Exception as e:
        print(
            f"Config fails at import/exec time (train.py would hit the same error): "
            f"{cfg_path}\n{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"Config OK: {cfg_path}")
    return cfg_path


def _read_train_config_vars(config_name: str) -> dict:
    """
    Best-effort parse of `patzer/config/{config_name}.py` to extract simple constants.
    Avoids importing/execing arbitrary config code.
    """
    cfg_path = Path(__file__).parent / "patzer" / "config" / f"{config_name}.py"
    if not cfg_path.exists():
        return {}
    txt = cfg_path.read_text()

    out: dict[str, object] = {}
    m = re.search(r"(?m)^\s*out_dir\s*=\s*['\"]([^'\"]+)['\"]\s*$", txt)
    if m:
        out["out_dir"] = m.group(1).strip()
    m = re.search(r"(?m)^\s*eval_interval\s*=\s*(\d+)\s*$", txt)
    if m:
        out["eval_interval"] = int(m.group(1))
    m = re.search(r"(?m)^\s*always_save_checkpoint\s*=\s*(True|False)\s*$", txt)
    if m:
        out["always_save_checkpoint"] = (m.group(1) == "True")
    m = re.search(r"(?m)^\s*ckpt_save_interval\s*=\s*(\d+)\s*$", txt)
    if m:
        out["ckpt_save_interval"] = int(m.group(1))
    m = re.search(r"(?m)^\s*ckpt_best_cooldown_steps\s*=\s*(\d+)\s*$", txt)
    if m:
        out["ckpt_best_cooldown_steps"] = int(m.group(1))
    return out


def _estimate_bandwidth_gb_per_hr(
    *,
    config_name: str,
    mins_per_1k_steps: float,
    full_ckpt_gb: float,
    weights_gb: float,
    improve_rate_per_eval: float,
    training_data_gb: float,
    amortize_download_over_hours: float,
    down_gb_per_hr_override: float | None,
    up_gb_per_hr_override: float | None,
) -> tuple[float, float, dict]:
    """
    Estimate upload/download GB/hr for Vast bandwidth costing.

    Upload model aligned with ``train.py`` R2 behavior:

    - ``ckpt.pt``: uploaded only when a latest checkpoint is written — at most every
      ``ckpt_save_interval`` steps (``0`` = every eval), not every eval by default.
    - ``weights_best.pt``: R2 upload is rate-limited by ``ckpt_best_cooldown_steps``
      (``0`` = no cooldown). Uses ``min(improve_rate×evals/hr, steps/hr÷cooldown)``.
      Does not model ``ckpt_best_min_delta`` (may slightly overestimate weights egress).

    Download: one-time ``data/<dataset>/`` pull; amortized over ``--amortize-download-over-hours``
    (default **24** for ~day-scale runs — not a burst rate).
    """
    cfg = _read_train_config_vars(config_name)

    if up_gb_per_hr_override is not None:
        up = max(0.0, float(up_gb_per_hr_override))
        up_full = up_wts = None
    else:
        eval_interval = int(cfg.get("eval_interval") or 1000)
        always_save = bool(cfg.get("always_save_checkpoint") if "always_save_checkpoint" in cfg else True)
        ckpt_save_interval = int(cfg.get("ckpt_save_interval", 0))
        ckpt_best_cooldown_steps = int(cfg.get("ckpt_best_cooldown_steps", 0))

        steps_per_hr = 60.0 * 1000.0 / max(1e-9, float(mins_per_1k_steps))
        evals_per_hr = steps_per_hr / float(eval_interval)

        if always_save:
            if ckpt_save_interval == 0:
                full_per_hr = float(full_ckpt_gb) * evals_per_hr
            else:
                full_per_hr = float(full_ckpt_gb) * steps_per_hr / float(ckpt_save_interval)
        else:
            full_per_hr = 0.0

        raw_weight_uploads_per_hr = max(0.0, float(improve_rate_per_eval)) * evals_per_hr
        if ckpt_best_cooldown_steps > 0:
            cap = steps_per_hr / float(ckpt_best_cooldown_steps)
            weight_uploads_per_hr = min(raw_weight_uploads_per_hr, cap)
        else:
            weight_uploads_per_hr = raw_weight_uploads_per_hr

        up_wts = float(weights_gb) * weight_uploads_per_hr
        up_full = full_per_hr
        up = up_full + up_wts

    if down_gb_per_hr_override is not None:
        down = max(0.0, float(down_gb_per_hr_override))
    else:
        denom = float(amortize_download_over_hours)
        down = float(training_data_gb) / denom if denom > 0 else 0.0

    meta = {
        "mins_per_1k_steps": float(mins_per_1k_steps),
        "full_ckpt_gb": float(full_ckpt_gb),
        "weights_gb": float(weights_gb),
        "improve_rate_per_eval": float(improve_rate_per_eval),
        "training_data_gb": float(training_data_gb),
        "amortize_download_over_hours": float(amortize_download_over_hours),
        "config_vars": cfg,
        "upload_full_ckpt_gb_per_hr": up_full,
        "upload_weights_gb_per_hr": up_wts,
        "used_overrides": {
            "up_gb_per_hr": up_gb_per_hr_override is not None,
            "down_gb_per_hr": down_gb_per_hr_override is not None,
        },
    }
    return up, down, meta


def list_instances():
    instances = vast("show", "instances")
    if not instances:
        print("No running instances.")
        return
    print(f"\n  {'ID':>10}  {'Status':<14}  {'GPU':<24}  {'$/hr':>6}")
    print("  " + "-" * 60)
    for inst in instances:
        print(f"  {inst['id']:>10}  {inst.get('actual_status','?'):<14}  "
              f"{inst.get('gpu_name','?'):<24}  ${inst.get('dph_total', 0):.3f}")
    print()


def ssh_prefix(info: dict) -> list[str]:
    host = info.get("ssh_host") or info.get("public_ipaddr")
    port = info.get("ssh_port", 22)
    return ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15", f"root@{host}"]


def show_status(instance_id: int):
    print(f"Fetching status for instance {instance_id}...")
    try:
        info = vast("show", "instance", str(instance_id))
    except RuntimeError as e:
        print(f"Could not fetch instance: {e}")
        return

    status = info.get("actual_status", "unknown")
    gpu = info.get("gpu_name", "?")
    cost = info.get("dph_total", 0)
    host = info.get("ssh_host") or info.get("public_ipaddr", "")
    port = info.get("ssh_port", 22)

    print(f"\n  Instance:  {instance_id}")
    print(f"  Status:    {status}")
    print(f"  GPU:       {gpu}")
    print(f"  Cost:      ${cost:.3f}/hr")
    if host:
        print(f"  SSH:       ssh -p {port} root@{host}")

    if status != "running" or not host:
        return

    print(f"\n--- Last 30 lines of /workspace/train.log ---")
    result = subprocess.run(
        ssh_prefix(info) + ["tail -n 30 /workspace/train.log 2>/dev/null || echo '(log not found)'"],
        capture_output=True, text=True,
    )
    print(result.stdout or result.stderr or "(no output)")


def run_on_instance(
    instance_id: int,
    config: str,
    resume: bool,
    full_reset: bool = False,
    terminate_on_error: bool = False,
    wandb_run_id: str | None = None,
):
    print(f"Fetching info for instance {instance_id}...")
    info = vast("show", "instance", str(instance_id))
    status = info.get("actual_status", "unknown")
    if status != "running":
        print(f"Instance {instance_id} is not running (status: {status}). Start it first.")
        sys.exit(1)

    host = info.get("ssh_host") or info.get("public_ipaddr")
    port = info.get("ssh_port", 22)
    print(f"Instance is running — {info.get('gpu_name', '?')} @ {host}:{port}")
    if full_reset:
        print(f"FULL RESET: wiping {WORKSPACE} and doing a fresh clone...")
    else:
        print(f"Updating {WORKSPACE} in-place (keeps data/)...")

    script = build_script(
        config,
        resume,
        wipe=full_reset,
        terminate_on_error=terminate_on_error,
        wandb_run_id=wandb_run_id,
    )

    # Write the script to the instance then run it in a tmux session.
    # Piping via stdin avoids any quoting nightmares with the script content.
    setup_cmd = (
        "apt-get install -y tmux -qq 2>/dev/null; "
        "cat > /tmp/patzer_train.sh; "
        "chmod +x /tmp/patzer_train.sh; "
        "tmux kill-session -t patzer 2>/dev/null || true; "
        "tmux new-session -d -s patzer "
        "  'bash /tmp/patzer_train.sh 2>&1 | tee /workspace/train.log'; "
        "echo 'Training started in tmux session: patzer'"
    )

    result = subprocess.run(
        ssh_prefix(info) + [setup_cmd],
        input=script,
        text=True,
    )
    if result.returncode != 0:
        print("Failed to start training on instance. Check SSH access.")
        sys.exit(1)

    print(f"""
Training started on instance {instance_id}.

  SSH:         ssh -p {port} root@{host}
  Attach tmux: ssh -p {port} root@{host} -t tmux attach -t patzer
  Tail log:    ssh -p {port} root@{host} tail -f /workspace/train.log
  Vast logs:   vastai logs {instance_id}
  Destroy:     vastai destroy instance {instance_id}
""")


def _legacy_prepend_train(argv: list[str]) -> None:
    """Support ``launch.py --config train_patzer_v4`` without the ``train`` keyword."""
    if len(argv) <= 1:
        return
    first = argv[1]
    if first == "train":
        return
    # Removed ``scrape`` subcommand — don't prepend ``train`` so argparse fails clearly.
    if first == "scrape":
        return
    # Top-level-only flags (must not insert ``train``, or ``--help`` shows the wrong parser).
    if first in ("-h", "--help", "--list"):
        return
    if first == "--status":
        return
    argv.insert(1, "train")


def cmd_train(args: argparse.Namespace) -> None:
    # Fail fast before SSH / vast create if config is missing or broken.
    if args.config.endswith(".py"):
        args.config = args.config[:-3]
    validate_train_config(args.config)

    if args.instance:
        run_on_instance(
            args.instance,
            args.config,
            args.resume,
            args.full_reset,
            args.terminate_on_error,
            args.wandb_run_id or None,
        )
        return

    # --- Rent a new instance ---

    if args.max_compute_cap < 0:
        print("--max-compute-cap must be >= 0", file=sys.stderr)
        sys.exit(2)
    if args.max_compute_cap > 0 and args.max_compute_cap < 750:
        print(
            "--max-compute-cap must be >= 750 (search already uses compute_cap>=750), "
            f"or 0 to disable the upper bound; got {args.max_compute_cap}",
            file=sys.stderr,
        )
        sys.exit(2)

    query = (
        f"num_gpus=1 gpu_ram>={args.min_gpu_ram} rentable=true verified=true "
        f"compute_cap>=750"
    )
    if args.max_compute_cap > 0:
        query += f" compute_cap<={args.max_compute_cap}"
    query += f" dph_total<={args.max_price}"
    if args.min_cuda_vers and args.min_cuda_vers > 0:
        query += f" cuda_vers>={args.min_cuda_vers}"
    if not args.allow_sliced_gpus:
        query += " gpu_frac=1"
    # Default 0 = free bandwidth only; negative disables the filter for that direction.
    if args.max_inet_up_cost >= 0:
        query += f" inet_up_cost<={args.max_inet_up_cost}"
    if args.max_inet_down_cost >= 0:
        query += f" inet_down_cost<={args.max_inet_down_cost}"
    if args.gpu_name:
        # Vast query strings: spaces in gpu_name must be underscores (e.g. RTX_4090).
        gn = str(args.gpu_name).strip().replace(" ", "_")
        query += f" gpu_name={gn}"

    search_args = ["search", "offers", query, "-o", "dph_total", "--limit", str(args.limit)]
    if args.interruptible:
        search_args.append("--interruptible")

    print(f"Searching: {query}")
    offers = vast(*search_args)

    if not offers:
        print(
            "No offers found. Try --max-price, --min-gpu-ram, "
            "--max-price / --gpu-name, "
            "or --max-inet-up-cost 0 --max-inet-down-cost 0 for free bandwidth only, "
            "or --search-only to explore.",
        )
        sys.exit(1)

    # Add an "all-in" $/hr estimate that includes expected bandwidth costs.
    up_gb_per_hr, down_gb_per_hr, bw_meta = _estimate_bandwidth_gb_per_hr(
        config_name=args.config,
        mins_per_1k_steps=args.mins_per_1k_steps,
        full_ckpt_gb=args.full_ckpt_gb,
        weights_gb=args.weights_gb,
        improve_rate_per_eval=args.improve_rate_per_eval,
        training_data_gb=args.training_data_gb,
        amortize_download_over_hours=args.amortize_download_over_hours,
        down_gb_per_hr_override=args.down_gb_per_hr,
        up_gb_per_hr_override=args.up_gb_per_hr,
    )
    cfg_vars = bw_meta.get("config_vars", {}) or {}
    eval_interval = cfg_vars.get("eval_interval", 1000)
    upl_full = bw_meta.get("upload_full_ckpt_gb_per_hr")
    upl_w = bw_meta.get("upload_weights_gb_per_hr")
    detail = ""
    if upl_full is not None and upl_w is not None:
        detail = f"  ({upl_full:.3f} ckpt.pt + {upl_w:.3f} weights_best R2)"
    cs = cfg_vars.get("ckpt_save_interval", 0)
    cd = cfg_vars.get("ckpt_best_cooldown_steps", 0)
    print(
        f"\nBandwidth assumptions for all-in $/hr:"
        f"\n  upload: {up_gb_per_hr:.3f} GB/hr {detail}"
        f"\n          sizes: ckpt.pt={args.full_ckpt_gb} GB, weights_best={args.weights_gb} GB | "
        f"improve_rate/eval={args.improve_rate_per_eval:.3f} | "
        f"mins/1k_steps={args.mins_per_1k_steps} eval_interval={eval_interval} "
        f"ckpt_save_interval={cs} cooldown_steps={cd} "
        f"always_save={cfg_vars.get('always_save_checkpoint', True)}"
        f"\n  download: {down_gb_per_hr:.3f} GB/hr  (training_data_gb={args.training_data_gb} "
        f"amortized over {args.amortize_download_over_hours} hr — one-time dataset pull)"
    )

    for o in offers:
        o["_all_in_dph"] = _offer_all_in_dph(
            o,
            up_gb_per_hr=up_gb_per_hr,
            down_gb_per_hr=down_gb_per_hr,
        )
    offers.sort(key=lambda x: (x.get("_all_in_dph", float("inf")), x.get("dph_total", float("inf"))))

    print_offers(offers)

    if args.search_only:
        return

    best = offers[0]
    print(
        f"\nDefault: #{1} — {best['gpu_name']} "
        f"@ base ${best['dph_total']:.3f}/hr, all-in ${best.get('_all_in_dph', best['dph_total']):.3f}/hr "
        f"(id={best['id']})"
    )
    print("Press Enter to use it, type a number to pick a different one, or Ctrl-C to abort.")
    try:
        choice = input("> ").strip()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)

    if choice == "":
        offer = best
    elif choice.isdigit() and 1 <= int(choice) <= len(offers):
        offer = offers[int(choice) - 1]
    else:
        print("Invalid choice. Aborted.")
        sys.exit(1)

    print(f"\nLaunching: {offer['gpu_name']} (id={offer['id']}) "
          f"@ ${offer['dph_total']:.3f}/hr  |  config: {args.config}"
          + ("  [RESUME]" if args.resume else ""))

    create_args = [
        "create", "instance", str(offer["id"]),
        "--image", IMAGE,
        "--disk", str(args.disk),
        "--ssh",
        "--env", r2_env_flags(),
        "--onstart-cmd", build_script(
            args.config,
            args.resume,
            wipe=False,
            terminate_on_error=args.terminate_on_error,
            wandb_run_id=args.wandb_run_id or None,
        ),
    ]
    if args.interruptible:
        bid = round(offer.get("min_bid", offer["dph_total"]) * 1.15, 4)
        create_args += ["--bid_price", str(bid)]

    result = vast(*create_args)
    instance_id = result["new_contract"]
    print(f"Instance created: {instance_id}")

    print("Waiting for instance to come online", end="", flush=True)
    info = {}
    for _ in range(60):
        time.sleep(10)
        try:
            info = vast("show", "instance", str(instance_id))
        except Exception:
            print(".", end="", flush=True)
            continue
        status = info.get("actual_status", "unknown")
        print(f" [{status}]", end="", flush=True)
        if status == "running":
            print()
            break
    else:
        print(f"\nTimed out. Check status with: vastai show instance {instance_id}")
        sys.exit(1)

    host = info.get("ssh_host") or info.get("public_ipaddr", "<host>")
    port = info.get("ssh_port", 22)

    print(f"""
Instance {instance_id} is running!

  SSH:         ssh -p {port} root@{host}
  Attach tmux: ssh -p {port} root@{host} -t tmux attach -t patzer
  Tail log:    ssh -p {port} root@{host} tail -f /workspace/train.log
  Vast logs:   vastai logs {instance_id}
  Stop:        vastai stop instance {instance_id}
  Destroy:     vastai destroy instance {instance_id}

Training is starting via the onstart script (git clone → pip install → train).
Checkpoints will push to R2 at every eval interval.
""")


def main():
    _legacy_prepend_train(sys.argv)

    parser = argparse.ArgumentParser(description="Patzer Vast.ai GPU training launcher.")
    parser.add_argument("--list", action="store_true",
                        help="List your running instances and exit")
    parser.add_argument("--status", type=int, metavar="ID", default=None,
                        help="Show status and recent training log for an instance")

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND", required=False)

    train_p = sub.add_parser("train", help="Rent a GPU and run patzer training")
    train_p.add_argument("--config", required=True,
                         help="Config filename without .py (e.g. train_patzer_v3)")
    train_p.add_argument("--resume", action="store_true",
                         help="Pull R2 checkpoint before training (pass init_from=resume)")
    train_p.add_argument(
        "--wandb-run-id",
        default="",
        help="W&B run id to resume into (only used with --resume). Example: a41yuxdq",
    )
    train_p.add_argument("--instance", type=int, metavar="ID",
                         help="Run on an existing instance instead of renting a new one")
    train_p.add_argument(
        "--full-reset",
        action="store_true",
        help=(
            "On an existing instance, wipe /workspace/patzer and reclone. "
            "Default is in-place git update (keeps data/ and avoids re-downloading train.bin)."
        ),
    )
    train_p.add_argument("--disk", type=int, default=40,
                         help="Disk size in GB for new instances (default: 40)")
    train_p.add_argument("--min-gpu-ram", type=int, default=8,
                         help="Min GPU VRAM in GB (default: 8)")
    train_p.add_argument(
        "--max-price",
        type=float,
        default=0.60,
        help=(
            "Max base $/hr filter (dph_total). Default 0.60 excludes most RTX 4090 class "
            "offers — try e.g. 1.2–2.5 with --search-only to browse."
        ),
    )
    train_p.add_argument(
        "--gpu-name",
        metavar="MODEL",
        default="",
        help=(
            "Restrict search to one GPU model (Vast gpu_name field). "
            'Examples: RTX_4090, \"RTX 4090\". Spaces are converted to underscores.'
        ),
    )
    train_p.add_argument(
        "--up-gb-per-hr",
        type=float,
        default=None,
        help="Override expected upload volume (GB/hr) for all-in $/hr (default: auto-estimate)",
    )
    train_p.add_argument(
        "--down-gb-per-hr",
        type=float,
        default=None,
        help="Override expected download volume (GB/hr) for all-in $/hr (default: amortized training data download)",
    )
    train_p.add_argument(
        "--full-ckpt-gb",
        type=float,
        default=2.5,
        dest="full_ckpt_gb",
        metavar="GB",
        help=(
            "Full checkpoint ckpt.pt size in GB (default: %(default)s). "
            "Upload rate follows config ckpt_save_interval (0 = every eval)."
        ),
    )
    train_p.add_argument(
        "--weights-gb",
        type=float,
        default=1.0,
        dest="weights_gb",
        metavar="GB",
        help=(
            "weights_best.pt size in GB (default: %(default)s). "
            "Upload rate is min(improve_rate×evals/hr, steps/hr/cooldown_steps) from config."
        ),
    )
    train_p.add_argument(
        "--mins-per-1k-steps",
        type=float,
        default=10.0,
        help="Training speed in minutes per 1000 steps (default: %(default)s)",
    )
    train_p.add_argument(
        "--improve-rate-per-eval",
        type=float,
        default=0.861,
        help="Probability an eval is a new best (default: 0.861 from recent v3 run)",
    )
    train_p.add_argument(
        "--training-data-gb",
        type=float,
        default=5.0,
        help="One-time train.bin (+ sidecars) download size in GB (default: %(default)s)",
    )
    train_p.add_argument(
        "--amortize-download-over-hours",
        type=float,
        default=24.0,
        help=(
            "Amortize one-time training-data download over N hours for steady-state $/hr "
            "(default: %(default)s — tuned for ~24h rental runs)"
        ),
    )
    train_p.add_argument(
        "--max-inet-up-cost",
        type=float,
        default=-1.0,
        metavar="$/GB",
        help=(
            "Require inet_up_cost<=this ($/GB upload). Default %(default)s = no filter. "
            "Use 0 for free upload only."
        ),
    )
    train_p.add_argument(
        "--max-inet-down-cost",
        type=float,
        default=-1.0,
        metavar="$/GB",
        help=(
            "Require inet_down_cost<=this ($/GB download). Default %(default)s = no filter. "
            "Use 0 for free download only."
        ),
    )
    train_p.add_argument("--search-only", action="store_true",
                         help="Print available offers and exit")
    train_p.add_argument(
        "--limit",
        type=int,
        default=10,
        help=(
            "Max offers fetched from Vast (sorted ascending by base $/hr). "
            "Default 10 is almost always cheap low-end GPUs; use a larger limit or "
            "--gpu-name RTX_4090 to surface specific models."
        ),
    )
    train_p.add_argument("--interruptible", "-i", action="store_true",
                         help="Use spot/interruptible pricing (~50%% cheaper, can be preempted)")
    train_p.add_argument(
        "--min-cuda-vers",
        type=float,
        default=12.4,
        metavar="VERS",
        help=(
            "Minimum Vast offer cuda_vers (host max CUDA from driver). "
            "Default 12.4 matches pytorch+cu12.4 images; use 0 to disable."
        ),
    )
    train_p.add_argument(
        "--max-compute-cap",
        type=int,
        default=DEFAULT_MAX_COMPUTE_CAP,
        metavar="CAP",
        help=(
            "Maximum Vast compute_cap (CUDA capability × 100). Default %(default)s matches "
            "PyTorch 2.5.x+cu12.4 prebuilts (through sm_90) and excludes Blackwell sm_120 (~1200). "
            "Use 0 to disable (e.g. after upgrading IMAGE / PyTorch for sm_120)."
        ),
    )
    train_p.add_argument(
        "--allow-sliced-gpus",
        action="store_true",
        help=(
            "Allow offers that are 1 GPU from a multi-GPU machine (gpu_frac<1). "
            "By default we filter to single-GPU machines (gpu_frac=1)."
        ),
    )
    train_p.add_argument("--terminate-on-error", action="store_true",
                         help="Destroy instance even if training fails (default: keep alive for debugging)")

    args = parser.parse_args()

    if args.list:
        list_instances()
        return

    if args.status is not None:
        show_status(args.status)
        return

    if not args.cmd:
        parser.error("specify the train command (see --help). "
                     "Example: python launch.py train --config train_patzer_v4")

    if args.cmd == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
