"""Aggregate gate LR ablation runs and verify the multiplier reached the optimizer."""

import argparse
import json
import re
import statistics
from pathlib import Path


def accuracy_from_log(run_dir, subset):
    text = (run_dir / f"test_{subset}" / "log.txt").read_text()
    values = re.findall(r"\* accuracy:\s*([0-9.]+)%", text)
    if len(values) != 1:
        raise RuntimeError(f"Expected one evaluation in {run_dir}/test_{subset}")
    return float(values[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    rows = []
    for dataset_dir in sorted(args.base_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for mode_dir in sorted(dataset_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            for seed_dir in sorted(mode_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                metrics = json.loads((seed_dir / "train_metrics.json").read_text())
                if metrics["status"] != "success":
                    raise RuntimeError(f"Training did not succeed: {seed_dir}")
                base_lr = metrics["base_lr"]
                multiplier = metrics["multiplier"]
                groups = metrics["optimizer_param_groups"]
                gate_groups = [
                    group for group in groups
                    if abs(group["lr"] - base_lr * multiplier) < 1e-9
                ]
                if not gate_groups:
                    raise RuntimeError(f"Gate LR not applied in optimizer: {seed_dir}")
                base_acc = accuracy_from_log(seed_dir, "base")
                novel_acc = accuracy_from_log(seed_dir, "new")
                rows.append(
                    {
                        "dataset": dataset_dir.name,
                        "mode": mode_dir.name,
                        "multiplier": multiplier,
                        "seed": int(seed_dir.name.replace("seed", "")),
                        "base_accuracy": base_acc,
                        "novel_accuracy": novel_acc,
                        "harmonic_mean": 2 * base_acc * novel_acc / (base_acc + novel_acc),
                        "gate_lr": gate_groups[0]["lr"],
                        "trainable_parameters": metrics["trainable_parameters"],
                        "gate_parameters": metrics["gate_parameters"],
                        "training_seconds": metrics["trainer_train_seconds_including_final_eval"],
                    }
                )

    report = {"status": "success", "runs": rows, "aggregates": []}
    groups = {}
    for row in rows:
        key = (row["dataset"], row["multiplier"])
        groups.setdefault(key, []).append(row)
    for key in sorted(groups):
        dataset, multiplier = key
        runs = groups[key]
        report["aggregates"].append(
            {
                "dataset": dataset,
                "multiplier": multiplier,
                "base_mean": statistics.mean(r["base_accuracy"] for r in runs),
                "base_std": statistics.stdev(r["base_accuracy"] for r in runs),
                "novel_mean": statistics.mean(r["novel_accuracy"] for r in runs),
                "novel_std": statistics.stdev(r["novel_accuracy"] for r in runs),
                "hm_mean": statistics.mean(r["harmonic_mean"] for r in runs),
                "hm_std": statistics.stdev(r["harmonic_mean"] for r in runs),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
