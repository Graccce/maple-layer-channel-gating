"""Compare fixed-one channel runs against the matching none baselines.

The decisive check is bitwise equality of every checkpoint tensor shared by the
two checkpoints, excluding the gate_alphas and gate schema keys that only the
fixed-one (channel) run has. Accuracy is reported for readability only.
"""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--none-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    report = {"status": "failed", "comparisons": []}
    for seed in args.seeds:
        row = {"seed": seed, "status": "failed"}
        fixed_ckpt = (
            args.fixed_dir
            / f"seed{seed}"
            / "train_base"
            / "MultiModalPromptLearner"
            / f"model.pth.tar-{args.epoch}"
        )
        none_ckpt = (
            args.none_dir
            / f"seed{seed}"
            / "train_base"
            / "MultiModalPromptLearner"
            / f"model.pth.tar-{args.epoch}"
        )
        fixed_state = torch.load(fixed_ckpt, map_location="cpu")["state_dict"]
        none_state = torch.load(none_ckpt, map_location="cpu")["state_dict"]
        common = sorted(set(fixed_state) & set(none_state))
        unequal = []
        for key in common:
            left = fixed_state[key].float()
            right = none_state[key].float()
            if not torch.equal(left, right):
                unequal.append(
                    {"key": key, "max_abs_diff": (left - right).abs().max().item()}
                )
        fixed_results = json.loads((args.fixed_dir / f"seed{seed}" / "results.json").read_text())
        none_results = json.loads((args.none_dir / f"seed{seed}" / "results.json").read_text())
        row.update(
            {
                "common_tensor_count": len(common),
                "unequal_tensor_count": len(unequal),
                "unequal": unequal,
                "fixed_base": fixed_results["base_accuracy"],
                "fixed_novel": fixed_results["novel_accuracy"],
                "fixed_hm": fixed_results["harmonic_mean"],
                "none_base": none_results["base_accuracy"],
                "none_novel": none_results["novel_accuracy"],
                "none_hm": none_results["harmonic_mean"],
            }
        )
        row["status"] = "identical" if not unequal else "mismatch"
        report["comparisons"].append(row)

    all_identical = all(entry["unequal_tensor_count"] == 0 for entry in report["comparisons"])
    report["status"] = "success"
    report["conclusion"] = (
        "fixed-one channel matches none bitwise"
        if all_identical
        else "fixed-one channel differs from none"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
