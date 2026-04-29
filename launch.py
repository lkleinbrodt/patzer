"""
Launch a Vast.ai GPU instance for patzer training.

python launch.py                      # cheapest offer, default config, confirm prompt
python launch.py --search-only        # print offers and exit
python launch.py --config train_patzer_v1
python launch.py --resume             # pull R2 checkpoint and pass --init_from=resume
python launch.py --interruptible      # use spot pricing (~50% cheaper, can be interrupted)
python launch.py --max-price 0.30     # cap $/hr
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
    return " ".join(f"-e {k}={os.environ[k]}" for k in keys if os.environ.get(k))


def build_onstart(config: str, resume: bool) -> str:
    pull_ckpt = (
        f"python -c \"import sys; sys.path.insert(0,'patzer'); import r2; "
        f"r2.pull_dir('checkpoints', 'checkpoints')\" || true"
        if resume else ""
    )
    resume_flag = "--init_from=resume" if resume else ""
    return "\n".join(filter(None, [
        "#!/bin/bash",
        "set -e",
        # persist R2 env vars for SSH sessions
        "env | grep -E '^R2_' >> /etc/environment",
        f"git clone {GITHUB_REPO} {WORKSPACE}",
        f"cd {WORKSPACE}",
        "pip install -q -r requirements.txt",
        "cd patzer",
        pull_ckpt,
        f"python train.py config/{config}.py {resume_flag}".strip(),
    ]))


def print_offers(offers):
    print(f"\n  {'#':>3}  {'GPU':<28} {'VRAM':>6} {'$/hr':>6} {'Reliability':>11}  {'ID':>10}")
    print("  " + "-" * 68)
    for i, o in enumerate(offers):
        vram_gb = int(o['gpu_ram']) // 1024
        print(f"  {i+1:>3}  {o['gpu_name']:<28} {vram_gb:>5}G "
              f"{o['dph_total']:>6.3f}  {o['reliability2']:>11.3f}  {o['id']:>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="train_patzer",
                        help="Config filename without .py (default: train_patzer)")
    parser.add_argument("--resume", action="store_true",
                        help="Pull R2 checkpoint before training (pass init_from=resume)")
    parser.add_argument("--disk", type=int, default=40, help="Disk size in GB (default: 40)")
    parser.add_argument("--min-gpu-ram", type=int, default=8,
                        help="Min GPU VRAM in GB (default: 8)")
    parser.add_argument("--max-price", type=float, default=0.60,
                        help="Max $/hr (default: 0.60)")
    parser.add_argument("--search-only", action="store_true",
                        help="Print available offers and exit")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of offers to show (default: 10)")
    parser.add_argument("--interruptible", "-i", action="store_true",
                        help="Use spot/interruptible pricing (~50% cheaper, can be preempted)")
    args = parser.parse_args()

    query = (f"num_gpus=1 gpu_ram>={args.min_gpu_ram} rentable=true verified=true "
             f"compute_cap>=750 dph_total<={args.max_price}")

    search_args = ["search", "offers", query, "-o", "dph_total",
                   "--limit", str(args.limit)]
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
        "--ssh", "--direct",
        "--env", r2_env_flags(),
        "--onstart-cmd", build_onstart(args.config, args.resume),
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

    ssh_host = info.get("ssh_host") or info.get("public_ipaddr", "<host>")
    ssh_port = info.get("ssh_port", 22)

    print(f"""
Instance {instance_id} is running!

  SSH:      ssh -p {ssh_port} root@{ssh_host}
  Logs:     vastai logs {instance_id}
  Stop:     vastai stop instance {instance_id}
  Destroy:  vastai destroy instance {instance_id}

Training is starting via the onstart script (git clone → pip install → train).
Checkpoints will push to R2 at every eval interval.
""")


if __name__ == "__main__":
    main()
