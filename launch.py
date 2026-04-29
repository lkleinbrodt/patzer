"""
Launch a Vast.ai GPU instance for patzer training.

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
    return "\n".join(lines)


def build_script(config: str, resume: bool, wipe: bool = False, terminate_on_error: bool = False) -> str:
    pull_ckpt = (
        f"python -c \"import sys; sys.path.insert(0,'patzer'); import r2; "
        f"r2.pull_dir('checkpoints', 'checkpoints')\" || true"
        if resume else ""
    )
    resume_flag = "--init_from=resume" if resume else ""

    # Vast.ai injects $CONTAINER_ID inside the container — that's the instance ID.
    # We install vastai so the script can self-destruct when done.
    kill_on_error = "1" if terminate_on_error else "0"
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
    if [ $EXIT_CODE -eq 0 ] || [ {kill_on_error} -eq 1 ]; then
        echo "Destroying instance ${{CONTAINER_ID}}..."
        vastai --api-key "$VAST_API_KEY" destroy instance "$CONTAINER_ID" || true
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
        trap,
        "cd patzer",
        pull_ckpt,
        f"python -u train.py config/{config}.py {resume_flag} 2>&1 | tee /workspace/train.log".strip(),
    ]))


def print_offers(offers):
    print(f"\n  {'#':>3}  {'GPU':<28} {'VRAM':>6} {'$/hr':>6} {'Reliability':>11}  {'ID':>10}")
    print("  " + "-" * 68)
    for i, o in enumerate(offers):
        vram_gb = int(o['gpu_ram']) // 1024
        print(f"  {i+1:>3}  {o['gpu_name']:<28} {vram_gb:>5}G "
              f"{o['dph_total']:>6.3f}  {o['reliability2']:>11.3f}  {o['id']:>10}")


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


def run_on_instance(instance_id: int, config: str, resume: bool, terminate_on_error: bool = False):
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

    script = build_script(config, resume, wipe=True, terminate_on_error=terminate_on_error)

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
    parser.add_argument("--config", default="train_patzer",
                        help="Config filename without .py (default: train_patzer)")
    parser.add_argument("--resume", action="store_true",
                        help="Pull R2 checkpoint before training (pass init_from=resume)")
    parser.add_argument("--instance", type=int, metavar="ID",
                        help="Run on an existing instance instead of renting a new one")
    parser.add_argument("--list", action="store_true",
                        help="List your running instances and exit")
    parser.add_argument("--disk", type=int, default=40,
                        help="Disk size in GB for new instances (default: 40)")
    parser.add_argument("--min-gpu-ram", type=int, default=8,
                        help="Min GPU VRAM in GB (default: 8)")
    parser.add_argument("--max-price", type=float, default=0.60,
                        help="Max $/hr (default: 0.60)")
    parser.add_argument("--search-only", action="store_true",
                        help="Print available offers and exit")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of offers to show (default: 10)")
    parser.add_argument("--interruptible", "-i", action="store_true",
                        help="Use spot/interruptible pricing (~50%% cheaper, can be preempted)")
    parser.add_argument("--terminate-on-error", action="store_true",
                        help="Destroy instance even if training fails (default: keep alive for debugging)")
    args = parser.parse_args()

    if args.list:
        list_instances()
        return

    if args.instance:
        run_on_instance(args.instance, args.config, args.resume, args.terminate_on_error)
        return

    # --- Rent a new instance ---

    query = (f"num_gpus=1 gpu_ram>={args.min_gpu_ram} rentable=true verified=true "
             f"compute_cap>=750 dph_total<={args.max_price}")

    search_args = ["search", "offers", query, "-o", "dph_total", "--limit", str(args.limit)]
    if args.interruptible:
        search_args.append("--interruptible")

    print(f"Searching: {query}")
    offers = vast(*search_args)

    if not offers:
        print("No offers found. Try --max-price, --min-gpu-ram, or --search-only to explore.")
        sys.exit(1)

    print_offers(offers)

    if args.search_only:
        return

    best = offers[0]
    print(f"\nDefault: #{1} — {best['gpu_name']} @ ${best['dph_total']:.3f}/hr  (id={best['id']})")
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
        "--onstart-cmd", build_script(args.config, args.resume, wipe=False,
                                      terminate_on_error=args.terminate_on_error),
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
