"""
Launch a Vast.ai GPU instance for patzer training.

Before any rent or --instance SSH, launch.py verifies `patzer/config/<name>.py` exists,
compiles, and executes locally (exit 2 on failure). --list / --status skip this check.

Default search includes ``cuda_vers>=12.4`` so hosts report driver ABI compatible with our
CUDA 12.4 PyTorch image; override with ``--min-cuda-vers 0`` to disable.

python launch.py                          # rent cheapest offer, confirm prompt
python launch.py --search-only            # print offers and exit
python launch.py --list                   # show your running instances
python launch.py --instance 12345678      # train on an existing instance (wipe + fresh clone)
python launch.py --config train_patzer_v1
python launch.py --resume                 # pull R2 checkpoint and pass --init_from=resume
python launch.py --interruptible          # spot pricing (~50% cheaper, can be preempted)
python launch.py --max-price 0.30
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
WORKSPACE = "/workspace/patzer"
VAST_API_KEY_PATH = Path.home() / ".config" / "vastai" / "vast_api_key"


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
        f"rm -rf {WORKSPACE}" if wipe else "",
        f"git clone {GITHUB_REPO} {WORKSPACE}",
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
    return out


def _estimate_bandwidth_gb_per_hr(
    *,
    config_name: str,
    mins_per_1k_steps: float,
    ckpt_gb: float,
    improve_rate_per_eval: float,
    training_data_gb: float,
    amortize_download_over_hours: float,
    down_gb_per_hr_override: float | None,
    up_gb_per_hr_override: float | None,
) -> tuple[float, float, dict]:
    """
    Estimate upload/download GB/hr for Vast bandwidth costing.

    Defaults are calibrated to our observed v3 behavior:
    - 7 min / 1k steps
    - eval_interval=1000
    - improvement rate ≈ 0.861 per eval (from W&B export)
    - ckpt_best checkpoint ≈ 0.5 GB

    We assume `always_save_checkpoint=True` => 1 checkpoint save per eval, plus
    a best-checkpoint save with probability=improve_rate_per_eval.
    """
    if up_gb_per_hr_override is not None:
        up = max(0.0, float(up_gb_per_hr_override))
    else:
        cfg = _read_train_config_vars(config_name)
        eval_interval = int(cfg.get("eval_interval") or 1000)
        always_save = bool(cfg.get("always_save_checkpoint") if "always_save_checkpoint" in cfg else True)

        steps_per_hr = 60.0 * 1000.0 / max(1e-9, float(mins_per_1k_steps))
        evals_per_hr = steps_per_hr / float(eval_interval)

        saves_per_eval = (1.0 if always_save else 0.0) + max(0.0, float(improve_rate_per_eval))
        up = float(ckpt_gb) * evals_per_hr * saves_per_eval

    if down_gb_per_hr_override is not None:
        down = max(0.0, float(down_gb_per_hr_override))
    else:
        denom = float(amortize_download_over_hours)
        down = float(training_data_gb) / denom if denom > 0 else 0.0

    meta = {
        "mins_per_1k_steps": float(mins_per_1k_steps),
        "ckpt_gb": float(ckpt_gb),
        "improve_rate_per_eval": float(improve_rate_per_eval),
        "training_data_gb": float(training_data_gb),
        "amortize_download_over_hours": float(amortize_download_over_hours),
        "config_vars": _read_train_config_vars(config_name),
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
    print(f"Wiping {WORKSPACE} and doing a fresh clone...")

    script = build_script(
        config,
        resume,
        wipe=True,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Config filename without .py (e.g. train_patzer_v3)")
    parser.add_argument("--resume", action="store_true",
                        help="Pull R2 checkpoint before training (pass init_from=resume)")
    parser.add_argument(
        "--wandb-run-id",
        default="",
        help="W&B run id to resume into (only used with --resume). Example: a41yuxdq",
    )
    parser.add_argument("--instance", type=int, metavar="ID",
                        help="Run on an existing instance instead of renting a new one")
    parser.add_argument("--list", action="store_true",
                        help="List your running instances and exit")
    parser.add_argument("--status", type=int, metavar="ID",
                        help="Show status and recent log for a running instance")
    parser.add_argument("--disk", type=int, default=40,
                        help="Disk size in GB for new instances (default: 40)")
    parser.add_argument("--min-gpu-ram", type=int, default=8,
                        help="Min GPU VRAM in GB (default: 8)")
    parser.add_argument("--max-price", type=float, default=0.60,
                        help="Max $/hr (default: 0.60)")
    parser.add_argument(
        "--up-gb-per-hr",
        type=float,
        default=None,
        help="Override expected upload volume (GB/hr) for all-in $/hr (default: auto-estimate)",
    )
    parser.add_argument(
        "--down-gb-per-hr",
        type=float,
        default=None,
        help="Override expected download volume (GB/hr) for all-in $/hr (default: amortized training data download)",
    )
    parser.add_argument(
        "--ckpt-gb",
        type=float,
        default=0.5,
        help="Checkpoint size in GB used for bandwidth estimate (default: 0.5)",
    )
    parser.add_argument(
        "--mins-per-1k-steps",
        type=float,
        default=7.0,
        help="Training speed in minutes per 1000 steps (default: 7.0)",
    )
    parser.add_argument(
        "--improve-rate-per-eval",
        type=float,
        default=0.861,
        help="Probability an eval is a new best (default: 0.861 from recent v3 run)",
    )
    parser.add_argument(
        "--training-data-gb",
        type=float,
        default=1.0,
        help="One-time training data download size in GB (default: 1.0)",
    )
    parser.add_argument(
        "--amortize-download-over-hours",
        type=float,
        default=24.0,
        help="Amortize training-data download over N hours when estimating down GB/hr (default: 24)",
    )
    parser.add_argument(
        "--max-inet-up-cost",
        type=float,
        default=None,
        help="Filter offers by max upload bandwidth cost ($/GB). Default: no filter.",
    )
    parser.add_argument(
        "--max-inet-down-cost",
        type=float,
        default=None,
        help="Filter offers by max download bandwidth cost ($/GB). Default: no filter.",
    )
    parser.add_argument("--search-only", action="store_true",
                        help="Print available offers and exit")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of offers to show (default: 10)")
    parser.add_argument("--interruptible", "-i", action="store_true",
                        help="Use spot/interruptible pricing (~50%% cheaper, can be preempted)")
    parser.add_argument(
        "--min-cuda-vers",
        type=float,
        default=12.4,
        metavar="VERS",
        help=(
            "Minimum Vast offer cuda_vers (host max CUDA from driver). "
            "Default 12.4 matches pytorch+cu12.4 images; use 0 to disable."
        ),
    )
    parser.add_argument(
        "--allow-sliced-gpus",
        action="store_true",
        help=(
            "Allow offers that are 1 GPU from a multi-GPU machine (gpu_frac<1). "
            "By default we filter to single-GPU machines (gpu_frac=1)."
        ),
    )
    parser.add_argument("--terminate-on-error", action="store_true",
                        help="Destroy instance even if training fails (default: keep alive for debugging)")
    args = parser.parse_args()

    if args.list:
        list_instances()
        return

    if args.status:
        show_status(args.status)
        return

    # Fail fast before SSH / vast create if config is missing or broken.
    if args.config.endswith(".py"):
        args.config = args.config[:-3]
    validate_train_config(args.config)

    if args.instance:
        run_on_instance(
            args.instance,
            args.config,
            args.resume,
            args.terminate_on_error,
            args.wandb_run_id or None,
        )
        return

    # --- Rent a new instance ---

    query = (
        f"num_gpus=1 gpu_ram>={args.min_gpu_ram} rentable=true verified=true "
        f"compute_cap>=750 dph_total<={args.max_price}"
    )
    if args.min_cuda_vers and args.min_cuda_vers > 0:
        query += f" cuda_vers>={args.min_cuda_vers}"
    if not args.allow_sliced_gpus:
        query += " gpu_frac=1"
    if args.max_inet_up_cost is not None:
        query += f" inet_up_cost<={args.max_inet_up_cost}"
    if args.max_inet_down_cost is not None:
        query += f" inet_down_cost<={args.max_inet_down_cost}"

    search_args = ["search", "offers", query, "-o", "dph_total", "--limit", str(args.limit)]
    if args.interruptible:
        search_args.append("--interruptible")

    print(f"Searching: {query}")
    offers = vast(*search_args)

    if not offers:
        print("No offers found. Try --max-price, --min-gpu-ram, or --search-only to explore.")
        sys.exit(1)

    # Add an "all-in" $/hr estimate that includes expected bandwidth costs.
    up_gb_per_hr, down_gb_per_hr, bw_meta = _estimate_bandwidth_gb_per_hr(
        config_name=args.config,
        mins_per_1k_steps=args.mins_per_1k_steps,
        ckpt_gb=args.ckpt_gb,
        improve_rate_per_eval=args.improve_rate_per_eval,
        training_data_gb=args.training_data_gb,
        amortize_download_over_hours=args.amortize_download_over_hours,
        down_gb_per_hr_override=args.down_gb_per_hr,
        up_gb_per_hr_override=args.up_gb_per_hr,
    )
    cfg_vars = bw_meta.get("config_vars", {}) or {}
    eval_interval = cfg_vars.get("eval_interval", 1000)
    always_save = cfg_vars.get("always_save_checkpoint", True)
    print(
        f"\nBandwidth assumptions for all-in $/hr:"
        f"\n  upload: {up_gb_per_hr:.3f} GB/hr  (ckpt_gb={args.ckpt_gb}, improve_rate/eval={args.improve_rate_per_eval:.3f}, "
        f"mins_per_1k_steps={args.mins_per_1k_steps}, eval_interval={eval_interval}, always_save_checkpoint={always_save})"
        f"\n  download: {down_gb_per_hr:.3f} GB/hr  (training_data_gb={args.training_data_gb} amortized over {args.amortize_download_over_hours} hr)"
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


if __name__ == "__main__":
    main()
