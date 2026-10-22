"""Combine per-epoch trajectory stats with per-epoch base/novel accuracy."""

import json
import re
import sys
from pathlib import Path


def accuracy_from_log(run_dir, subset, epoch):
    text = (run_dir / f"test_{subset}_epoch{epoch}" / "log.txt").read_text()
    values = re.findall(r"\* accuracy:\s*([0-9.]+)%", text)
    if len(values) != 1:
        raise RuntimeError(f"Expected one evaluation in {run_dir}/test_{subset}_epoch{epoch}")
    return float(values[0])


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/maple/summarize_trajectory.py RUN_DIR")
    run_dir = Path(sys.argv[1])
    trajectory = json.loads((run_dir / "trajectory.json").read_text())
    if trajectory["status"] != "success":
        raise SystemExit(f"Trajectory training did not succeed: {run_dir}")

    rows = []
    for entry in trajectory["epochs"]:
        epoch = entry["epoch"]
        base = accuracy_from_log(run_dir, "base", epoch)
        novel = accuracy_from_log(run_dir, "new", epoch)
        rows.append(
            {
                "epoch": epoch,
                "base_accuracy": base,
                "novel_accuracy": novel,
                "harmonic_mean": 2 * base * novel / (base + novel) if base + novel else 0,
                "train_loss_mean": entry["train_loss_mean"],
                "gate_grad_norm_epoch_l2": entry["gate_grad_norm_epoch_l2"],
                "other_grad_norm_epoch_l2": entry["other_grad_norm_epoch_l2"],
                "gate_layers": entry["gate_layers"],
                "lr_groups": entry["lr_groups"],
            }
        )
    summary = {"run": str(run_dir), "epochs": rows}
    out = run_dir / "trajectory_summary.json"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    with out.open("x") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
