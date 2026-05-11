"""
Estimate checkpoint improvement rate (new best val loss) from a W&B CSV export,
and convert it into an expected bandwidth egress rate for Vast offer selection.

Uses the same upload model as ``launch.py`` (see ``train.py``): ``ckpt.pt`` at
``ckpt_save_interval`` steps, ``weights_best.pt`` R2 uploads capped by cooldown.

Usage:
  python eval/estimate_vast_bandwidth.py /path/to/wandb_export.csv \\
      --full-ckpt-gb 2.5 --weights-gb 1.0 --ckpt-save-interval 10000 \\
      --cooldown-steps 2500 --mins-per-1k-steps 10 --eval-interval-steps 1000

Notes:
- Assumes CSV rows are in chronological order.
- Counts "improvement" when val/loss is strictly less than the best so far.
- Does not model ckpt_best_min_delta (R2 may upload slightly less often).
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
        default=10.0,
        help="Training speed in minutes per 1000 steps (default: %(default)s)",
    )
    ap.add_argument(
        "--eval-interval-steps",
        type=int,
        default=1000,
        help="Eval interval in steps (default: 1000)",
    )
    ap.add_argument(
        "--full-ckpt-gb",
        type=float,
        default=2.5,
        dest="full_ckpt_gb",
        help="Full ckpt.pt size in GB (default: 2.5)",
    )
    ap.add_argument(
        "--weights-gb",
        type=float,
        default=1.0,
        dest="weights_gb",
        help="weights_best.pt size in GB (default: 1.0)",
    )
    ap.add_argument(
        "--ckpt-save-interval",
        type=int,
        default=10000,
        dest="ckpt_save_interval",
        help="train.py ckpt_save_interval; 0 = save full ckpt every eval (default: 10000)",
    )
    ap.add_argument(
        "--cooldown-steps",
        type=int,
        default=2500,
        dest="cooldown_steps",
        help="train.py ckpt_best_cooldown_steps; 0 = no cooldown (default: 2500)",
    )
    args = ap.parse_args()

    if args.mins_per_1k_steps <= 0:
        raise SystemExit("--mins-per-1k-steps must be > 0")
    if args.eval_interval_steps <= 0:
        raise SystemExit("--eval-interval-steps must be > 0")
    if args.full_ckpt_gb < 0:
        raise SystemExit("--full-ckpt-gb must be >= 0")
    if args.weights_gb < 0:
        raise SystemExit("--weights-gb must be >= 0")
    if args.ckpt_save_interval < 0:
        raise SystemExit("--ckpt-save-interval must be >= 0")
    if args.cooldown_steps < 0:
        raise SystemExit("--cooldown-steps must be >= 0")

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

    if args.ckpt_save_interval == 0:
        full_per_hr = args.full_ckpt_gb * evals_per_hr
    else:
        full_per_hr = args.full_ckpt_gb * steps_per_hr / float(args.ckpt_save_interval)

    raw_u = improve_rate * evals_per_hr
    if args.cooldown_steps > 0:
        cap = steps_per_hr / float(args.cooldown_steps)
        weight_u_per_hr = min(raw_u, cap)
    else:
        weight_u_per_hr = raw_u

    weights_per_hr = args.weights_gb * weight_u_per_hr
    up_gb_per_hr = full_per_hr + weights_per_hr

    print(f"CSV: {args.csv_path}")
    print(f"Columns: step={step_col!r}, val={val_col!r}")
    print()
    print(f"Eval points:        {evals}")
    print(f"New-best events:    {improvements}")
    print(f"Improve rate/eval:  {improve_rate:.3f}")
    print()
    print(f"Speed:             {args.mins_per_1k_steps:.2f} min / 1k steps  ({steps_per_hr:.0f} steps/hr)")
    print(f"Eval interval:      {args.eval_interval_steps} steps  ({evals_per_hr:.2f} evals/hr)")
    print(f"ckpt_save_interval: {args.ckpt_save_interval}")
    print(f"cooldown_steps:     {args.cooldown_steps}")
    print()
    print(f"ckpt.pt:            {full_per_hr:.3f} GB/hr")
    print(f"weights_best R2:    {weights_per_hr:.3f} GB/hr  (min raw vs cooldown cap)")
    print(f"Total upload:       {up_gb_per_hr:.3f} GB/hr")
    print("Use with launch.py: --up-gb-per-hr {:.3f}".format(up_gb_per_hr))


if __name__ == "__main__":
    main()
