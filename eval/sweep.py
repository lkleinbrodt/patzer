"""
eval/sweep.py

Run a tournament for every checkpoint iteration, then plot score vs training step.

Discovers all checkpoints under a given R2 prefix, pulls them one at a time,
runs a short tournament, saves results to eval/results.json, then plots.

Usage:
    # Full sweep, delete each checkpoint after eval (saves disk):
    python eval/sweep.py \\
        --prefix checkpoints/patzer_v0 \\
        --depths 1 3 \\
        --games 20 \\
        --stockfish /opt/homebrew/bin/stockfish \\
        --conditioning match_color \\
        --device mps

    # Re-use cached local checkpoints, keep them after:
    python eval/sweep.py --prefix checkpoints/patzer_v0 --keep --depths 1

    # Just re-plot from existing results.json (no tournament):
    python eval/sweep.py --plot-only

Options:
    --skip-existing   Skip checkpoints that already have results with matching
                      (checkpoint_key, depth, conditioning). Default: on.
    --keep            Keep downloaded checkpoints on disk after eval (default: delete).
    --plot-only       Skip all downloading/playing; just regenerate the plot.
    --out             Output path for the plot (default: eval/sweep_plot.png).
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Checkpoint discovery ───────────────────────────────────────────────────────

def discover_checkpoints(r2_prefix: str) -> list[dict]:
    """
    List all .pt files under r2_prefix in R2, sorted by iter number.
    Returns list of dicts: {r2_key, filename, iter_hint}
    iter_hint is parsed from the filename (e.g. ckpt_005000.pt → 5000).
    ckpt.pt gets iter_hint=None (actual iter_num comes from the checkpoint itself).
    """
    from patzer.r2 import _client
    client, bucket = _client()
    if client is None:
        print("R2 not configured — set credentials in .env", file=sys.stderr)
        sys.exit(1)

    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=r2_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".pt"):
                keys.append(obj["Key"])

    entries = []
    for key in keys:
        fname = Path(key).name
        m = re.search(r"_(\d+)\.pt$", fname)
        iter_hint = int(m.group(1)) if m else None
        entries.append({"r2_key": key, "filename": fname, "iter_hint": iter_hint})

    # Sort: numbered checkpoints first by number, then ckpt.pt at the end
    def sort_key(e):
        return e["iter_hint"] if e["iter_hint"] is not None else float("inf")

    return sorted(entries, key=sort_key)


# ── Already-evaluated lookup ───────────────────────────────────────────────────

def already_evaluated(r2_key: str, depth: int, conditioning: str) -> bool:
    from eval.tournament import load_results
    for r in load_results():
        if (
            r.get("checkpoint") == r2_key
            and r.get("stockfish_depth") == depth
            and r.get("conditioning") == conditioning
        ):
            return True
    return False


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(args):
    import torch
    from patzer.r2 import pull_file
    from eval.engine import Patzer, StockfishPlayer
    from eval.tournament import play_game, save_results, run_tournament

    checkpoints = discover_checkpoints(args.prefix)
    if args.step:
        checkpoints = [
            e for e in checkpoints
            if e["iter_hint"] is None or e["iter_hint"] % args.step == 0
        ]
    print(f"Found {len(checkpoints)} checkpoint(s) in R2 under '{args.prefix}'"
          + (f" (step={args.step})" if args.step else ""))

    for entry in checkpoints:
        r2_key = entry["r2_key"]
        local_path = Path(r2_key)

        # Per-depth skip check
        depths_to_run = []
        for depth in args.depths:
            if args.skip_existing and already_evaluated(r2_key, depth, args.conditioning):
                print(f"  [skip] {entry['filename']} depth={depth} already in results.json")
            else:
                depths_to_run.append(depth)

        if not depths_to_run:
            continue

        # Pull from R2 if not local
        pulled = False
        if not local_path.exists():
            print(f"\n[r2] pulling {r2_key} ...")
            ok = pull_file(r2_key, local_path)
            if not ok:
                print(f"  [error] failed to pull {r2_key}, skipping", file=sys.stderr)
                continue
            pulled = True
        else:
            print(f"\n[local] {entry['filename']} already on disk")

        try:
            patzer = Patzer(
                local_path,
                device=args.device,
                temperature=args.temperature,
                top_k=args.top_k,
                conditioning=args.conditioning,
            )
            records = run_tournament(
                patzer,
                r2_key,  # store the R2 key as canonical checkpoint identifier
                args.stockfish,
                depths_to_run,
                args.games,
            )
            save_results(records)
        finally:
            # Clean up unless --keep or the file was already there before we pulled
            if pulled and not args.keep and local_path.exists():
                local_path.unlink()
                print(f"  [cleanup] deleted {local_path}")

    print("\nSweep complete.")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_sweep(out_path: Path, conditioning: str | None = None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib")
        return

    from eval.tournament import load_results, aggregate_results

    records = load_results()
    if not records:
        print("No results to plot.")
        return

    # Filter by conditioning if specified
    if conditioning:
        records = [r for r in records if r.get("conditioning") == conditioning]
    if not records:
        print(f"No results matching conditioning={conditioning}")
        return

    rows = aggregate_results(records)
    depths = sorted({r["stockfish_depth"] for r in rows})
    colors = ["#e15759", "#f28e2b", "#4e79a7", "#59a14f", "#76b7b2"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for depth, color in zip(depths, colors):
        depth_rows = [r for r in rows if r["stockfish_depth"] == depth]
        points = []
        for r in depth_rows:
            total = r["W"] + r["L"] + r["D"]
            score = (r["W"] + 0.5 * r["D"]) / total * 100 if total else 0
            points.append((r["iter_num"], score))

        if not points:
            continue

        points.sort()
        iters = [p[0] for p in points]
        scores = [p[1] for p in points]

        ax.plot(iters, scores, marker="o", color=color, linewidth=2,
                markersize=5, label=f"vs Stockfish depth {depth}")
        ax.fill_between(iters, scores, alpha=0.08, color=color)

    ax.set_xlabel("Training iteration", fontsize=12)
    ax.set_ylabel("Score % (W + ½D)", fontsize=12)
    title = "Patzer performance vs Stockfish over training"
    if conditioning:
        title += f"  [conditioning={conditioning}]"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[plot] saved → {out_path}")
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep all checkpoint iters and plot perf vs training step")
    parser.add_argument("--prefix", default="checkpoints/patzer_v0",
                        help="R2 prefix to search for checkpoints")
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 3],
                        help="Stockfish depths to evaluate against")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per checkpoint per depth")
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--conditioning", default="match_color",
                        help="Result conditioning token strategy")
    parser.add_argument("--step", type=int, default=None,
                        help="Only evaluate checkpoints whose iter is divisible by STEP "
                             "(e.g. --step 5000 evaluates at 5k, 10k, 15k, 20k). "
                             "ckpt.pt is always included regardless.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep downloaded checkpoints on disk (default: delete after eval)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip checkpoints already in results.json (default: on)")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false",
                        help="Re-run even if results already exist")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip tournament, just regenerate the plot from results.json")
    parser.add_argument("--out", default="eval/sweep_plot.png",
                        help="Output path for the plot")
    args = parser.parse_args()

    if not args.plot_only:
        run_sweep(args)

    plot_sweep(Path(args.out), conditioning=args.conditioning)


if __name__ == "__main__":
    main()
