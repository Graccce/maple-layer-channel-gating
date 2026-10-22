"""Per-seed result for a fixed-one channel run (gates frozen at identity)."""

import json
import re
import sys
from pathlib import Path


def accuracy_from_log(run_dir, subset):
    text = (run_dir / f"test_{subset}" / "log.txt").read_text()
    values = re.findall(r"\* accuracy:\s*([0-9.]+)%", text)
    if len(values) != 1:
        raise RuntimeError(f"Expected one evaluation in {run_dir}/test_{subset}")
    return float(values[0])


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/maple/summarize_fixedone.py RUN_DIR")
    run = Path(sys.argv[1])
    report = json.loads((run / "fixedone_report.json").read_text())
    if report["status"] != "success":
        raise RuntimeError(f"Training did not succeed: {run}")
    if not report.get("gates_verified_frozen_zero_after_training"):
        raise RuntimeError(f"Gates were not verified frozen at zero: {run}")
    base = accuracy_from_log(run, "base")
    novel = accuracy_from_log(run, "new")
    row = {
        "run": str(run),
        "base_accuracy": base,
        "novel_accuracy": novel,
        "harmonic_mean": 2 * base * novel / (base + novel),
        "status": "success",
        "gates_verified_frozen_zero_after_training": True,
    }
    out = run / "results.json"
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    with out.open("x") as handle:
        json.dump(row, handle, indent=2)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
