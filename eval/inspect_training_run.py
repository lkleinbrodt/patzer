"""Print wandb run metrics as CSV (iter, train_loss, val_loss, lr)."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import wandb

WANDB_RUNS_PREFIX = "lkleinbrodt-capital-group/patzer/runs"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_WANDB_RUNS_CSV_DIR = _REPO_ROOT / "data" / "wandb_runs"
WANDB_KEYS = ("iter", "train/loss", "val/loss", "lr")
CSV_FIELDS = ("iter", "train_loss", "val_loss", "lr")


def _cell(v: Any) -> str:
    if v is None or v == "":
        return ""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x.is_integer():
        return str(int(x))
    return f"{x:.8g}"


def _csv_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "iter": _cell(row.get("iter")),
        "train_loss": _cell(row.get("train/loss")),
        "val_loss": _cell(row.get("val/loss")),
        "lr": _cell(row.get("lr")),
    }


def _history_rows(
    run: Any,
    *,
    keys: tuple[str, ...],
    page_size: int,
) -> Iterator[Dict[str, Any]]:
    """Stream history rows; larger page_size means fewer W&B API round trips."""
    yield from run.scan_history(keys=list(keys), page_size=page_size)


def _run_id_clean(run_id: str) -> str:
    return run_id.strip().strip("/")


def _run_path(run_id: str) -> str:
    return f"{WANDB_RUNS_PREFIX}/{_run_id_clean(run_id)}"


def _default_output_csv(run_id: str) -> Path:
    rid = _run_id_clean(run_id).replace("/", "_")
    return _WANDB_RUNS_CSV_DIR / f"{rid}.csv"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            f"Example: %(prog)s abc123xyz  →  data/wandb_runs/abc123xyz.csv  "
            f"(W&B path {WANDB_RUNS_PREFIX}/<run_id>)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "run_id",
        help=f"W&B run id (full path is {WANDB_RUNS_PREFIX}/<run_id>)",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help=f"Output CSV (default: data/wandb_runs/<run_id>.csv under repo root)",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        metavar="N",
        help="Rows per W&B API page (larger = fewer requests; default 10000)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        metavar="N",
        help="Print row count to stderr every N rows (0 to disable)",
    )
    p.add_argument(
        "--echo",
        action="store_true",
        help="Print a short preview of the CSV to stdout after writing",
    )
    p.add_argument(
        "--echo-lines",
        type=int,
        default=25,
        metavar="N",
        help="With --echo, max lines to print from the start of the file",
    )
    args = p.parse_args()

    out_path = (
        Path(args.output) if args.output is not None else _default_output_csv(args.run_id)
    ).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_path = _run_path(args.run_id)
    api = wandb.Api()
    run = api.run(run_path)

    n = 0
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in _history_rows(run, keys=WANDB_KEYS, page_size=args.page_size):
            writer.writerow(_csv_row(row))
            n += 1
            pe = args.progress_every
            if pe and n % pe == 0:
                print(f"... {n} rows written", file=sys.stderr)

    if n == 0:
        print("No history rows returned.", file=sys.stderr)
        sys.exit(1)

    print(f"Training run {run_path} saved to {out_path} ({n} rows)", file=sys.stderr)

    if args.echo:
        print(f"CSV preview (first {args.echo_lines} lines):")
        with open(out_path, newline="") as f:
            for i, line in enumerate(f):
                if i >= args.echo_lines:
                    if n > args.echo_lines:
                        print(f"... ({n - args.echo_lines} more lines in file)")
                    break
                print(line, end="")


if __name__ == "__main__":
    main()
