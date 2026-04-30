"""Print wandb run metrics as CSV (iter, train_loss, val_loss, lr)."""

import csv
import sys
from typing import Any, Dict

import wandb

# Full run path: entity/project/runs/<id>
RUN_PATH = "/lkleinbrodt-capital-group/patzer/runs/fi1op0wk"

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


def main() -> None:
    api = wandb.Api()
    run = api.run(RUN_PATH)
    rows = list(run.scan_history(keys=list(WANDB_KEYS)))
    if not rows:
        print("No history rows returned.", file=sys.stderr)
        sys.exit(1)
    rows.sort(key=lambda r: float(r.get("iter") or 0))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(CSV_FIELDS))
    writer.writeheader()
    for row in rows:
        writer.writerow(_csv_row(row))


if __name__ == "__main__":
    main()
