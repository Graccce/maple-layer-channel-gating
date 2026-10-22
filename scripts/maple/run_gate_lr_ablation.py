"""Diagnostic: train with an independent learning-rate multiplier on gate_alphas.

Wraps the existing train.py and never converts or rewrites saved checkpoints.
The multiplier applies only to gate_alphas; every other trainable prompt
parameter keeps the configured LR. Because the MaPLe recipe uses a one-epoch
constant warmup, the multiplier takes effect after the warmup epoch.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import torch


def build_gate_lr_param_groups(model, optim_cfg, multiplier):
    """Split trainable params into (other, gate) groups for SGD.

    Returns a list suitable for torch.optim.SGD. The gate group carries an
    explicit per-group LR; the other group falls back to the configured LR.
    """
    gate = []
    other = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "gate_alphas" in name:
            gate.append(param)
        else:
            other.append(param)
    groups = [{"params": other, "lr": optim_cfg.LR}]
    if gate:
        groups.append({"params": gate, "lr": optim_cfg.LR * multiplier})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("report", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.multiplier <= 0:
        raise SystemExit("multiplier must be positive")
    if args.report.exists():
        raise SystemExit(f"Refusing to overwrite {args.report}")
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA; no experiment started")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import trainers.maple as maple_mod

    real_build_optimizer = maple_mod.build_optimizer
    captured = {}

    def patched_build_optimizer(model, optim_cfg):
        if isinstance(model, torch.nn.Module):
            has_gate = any(
                "gate_alphas" in name and param.requires_grad
                for name, param in model.named_parameters()
            )
            if has_gate:
                groups = build_gate_lr_param_groups(model, optim_cfg, args.multiplier)
                optim = real_build_optimizer(model, optim_cfg, param_groups=groups)
                captured["base_lr"] = optim_cfg.LR
                captured["multiplier"] = args.multiplier
                captured["param_groups"] = [
                    {"lr": group.get("lr", optim_cfg.LR), "n_params": len(group["params"])}
                    for group in optim.param_groups
                ]
                return optim
        return real_build_optimizer(model, optim_cfg)

    maple_mod.build_optimizer = patched_build_optimizer

    from trainers.maple import MaPLe

    metrics = {"multiplier": args.multiplier, "status": "failed"}
    original_train = MaPLe.train

    def measured_train(self, *train_args, **train_kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            return original_train(self, *train_args, **train_kwargs)
        finally:
            torch.cuda.synchronize()
            metrics["trainer_train_seconds_including_final_eval"] = time.perf_counter() - start
            metrics["trainable_parameters"] = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            metrics["gate_parameters"] = sum(
                p.numel() for n, p in self.model.named_parameters() if "gate_alphas" in n
            )

    MaPLe.train = measured_train
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    sys.argv = ["train.py", *args.train_args]
    try:
        runpy.run_path("train.py", run_name="__main__")
        metrics["status"] = "success"
    finally:
        torch.cuda.synchronize()
        metrics["process_seconds"] = time.perf_counter() - start
        metrics["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        metrics["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        metrics["argv"] = sys.argv.copy()
        if captured:
            metrics["optimizer_param_groups"] = captured["param_groups"]
            metrics["base_lr"] = captured["base_lr"]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(metrics, handle, indent=2)
        print("GATE_LR_ABLATION_REPORT", json.dumps(metrics))


if __name__ == "__main__":
    main()
