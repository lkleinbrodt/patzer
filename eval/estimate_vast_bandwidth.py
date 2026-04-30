"""
Estimate checkpoint improvement rate (new best val loss) from a W&B CSV export,
and convert it into an expected bandwidth egress rate for Vast offer selection.

Usage:
  python eval/estimate_vast_bandwidth.py /path/to/wandb_export.csv \
      --ckpt-gb 0.5 --mins-per-1k-steps 7 --eval-interval-steps 1000

Notes:
- Assumes CSV rows are in chronological order.
- Counts "improvement" when val/loss is strictly less than the best so far.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _find_col(fieldnames: list[str], want: str) -> str:
    # Prefer exact match, else substring match (case-insensitive).
    for f in fieldnames:
        if f == want:
            return f
    low = want.lower()
    for f in fieldnames:
        if low in f.lower():
            return f
    raise SystemExit(f"Could not find column matching {want!r}. Columns: {fieldnames}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path, help="W&B CSV export path")
    ap.add_argument("--step-col", default="Step", help="Step column name (default: Step)")
    ap.add_argument(
        "--val-col",
        default="val/loss",
        help="Val loss column name or substring to match (default: val/loss)",
    )
    ap.add_argument(
        "--mins-per-1k-steps",
        type=float,
        default=7.0,
        help="Training speed in minutes per 1000 steps (default: 7)",
    )
    ap.add_argument(
        "--eval-interval-steps",
        type=int,
        default=1000,
        help="Eval interval in steps (default: 1000)",
    )
    ap.add_argument(
        "--ckpt-gb",
        type=float,
        default=0.5,
        help="Checkpoint size in GB (default: 0.5)",
    )
    ap.add_argument(
        "--extra-saves-per-eval",
        type=float,
        default=1.0,
        help=(
            "Extra saves per eval besides the always-save checkpoint. "
            "If you save best checkpoints, this is approximately the 'improvement rate'. "
            "Leave at 1.0 to estimate '1 (always) + improve_rate'."
        ),
    )
    args = ap.parse_args()

    if args.mins_per_1k_steps <= 0:
        raise SystemExit("--mins-per-1k-steps must be > 0")
    if args.eval_interval_steps <= 0:
        raise SystemExit("--eval-interval-steps must be > 0")
    if args.ckpt_gb < 0:
        raise SystemExit("--ckpt-gb must be >= 0")

    with args.csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit("CSV has no header")

        step_col = _find_col(list(r.fieldnames), args.step_col)
        val_col = _find_col(list(r.fieldnames), args.val_col)

        rows = []
        for row in r:
            try:
                step = int(float(row[step_col]))
                val = float(row[val_col])
            except Exception:
                continue
            if math.isnan(val) or math.isinf(val):
                continue
            rows.append((step, val))

    if not rows:
        raise SystemExit("No usable rows found (check column names?)")

    rows.sort(key=lambda x: x[0])

    best = float("inf")
    improvements = 0
    for _, v in rows:
        if v < best:
            if best != float("inf"):
                improvements += 1
            best = v

    evals = len(rows)
    improve_rate = improvements / evals if evals else 0.0

    steps_per_hr = 60.0 * 1000.0 / args.mins_per_1k_steps
    evals_per_hr = steps_per_hr / float(args.eval_interval_steps)

    # Expected saves/hr: always_save_checkpoint=True means one per eval;
    # plus best checkpoint on improvement, approximated as improve_rate per eval.
    saves_per_eval = 1.0 + improve_rate * float(args.extra_saves_per_eval)
    saves_per_hr = evals_per_hr * saves_per_eval
    up_gb_per_hr = saves_per_hr * args.ckpt_gb

    print(f"CSV: {args.csv_path}")
    print(f"Columns: step={step_col!r}, val={val_col!r}")
    print()
    print(f"Eval points:        {evals}")
    print(f"New-best events:    {improvements}")
    print(f"Improve rate/eval:  {improve_rate:.3f}")
    print()
    print(f"Speed:             {args.mins_per_1k_steps:.2f} min / 1k steps  ({steps_per_hr:.0f} steps/hr)")
    print(f"Eval interval:      {args.eval_interval_steps} steps  ({evals_per_hr:.2f} evals/hr)")
    print(f"Checkpoint size:    {args.ckpt_gb:.3f} GB")
    print(f"Expected saves/eval {saves_per_eval:.3f}  (1 always + {improve_rate:.3f} improvement)")
    print()
    print(f"Estimated upload:   {up_gb_per_hr:.3f} GB/hr")
    print("Use with launch.py: --up-gb-per-hr {:.3f}".format(up_gb_per_hr))


if __name__ == "__main__":
    main()

