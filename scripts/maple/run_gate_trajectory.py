"""Diagnostic: record per-epoch gate and optimization trajectory for one run.

Wraps train.py without changing the optimization. Requires GATE_TYPE in
{scalar, channel}. Hooks before_epoch, forward_backward and after_epoch to
record gate statistics, gradient norms and training loss. Set
TRAIN.CHECKPOINT_FREQ 1 on the command line to keep per-epoch checkpoints for
later evaluation.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys

import torch


def gate_layer_summary(learner):
    rows = []
    for index, alpha in enumerate(learner.gate_alphas):
        value = alpha.detach().float()
        gate = 1.0 + 0.5 * torch.tanh(value)
        rows.append(
            {
                "layer": index,
                "numel": value.numel(),
                "alpha_l2": value.norm().item(),
                "alpha_max_abs": value.abs().max().item(),
                "gate_mean": gate.mean().item(),
                "gate_min": gate.min().item(),
                "gate_max": gate.max().item(),
                "max_abs_deviation_from_one": (gate - 1.0).abs().max().item(),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"Refusing to overwrite {args.report}")
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA; no experiment started")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from trainers.maple import MaPLe

    original_before_epoch = MaPLe.before_epoch
    original_after_epoch = MaPLe.after_epoch
    original_forward_backward = MaPLe.forward_backward

    records = {"status": "failed", "epochs": []}

    def before_epoch(self):
        self._diag_loss_sum = 0.0
        self._diag_n_batches = 0
        self._diag_gate_grad_sq = 0.0
        self._diag_other_grad_sq = 0.0
        return original_before_epoch(self)

    def forward_backward(self, batch):
        summary = original_forward_backward(self, batch)
        loss = float(summary["loss"])
        self._diag_loss_sum += loss
        self._diag_n_batches += 1
        gate_sq = 0.0
        other_sq = 0.0
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            norm = param.grad.detach().float().norm().item()
            if "gate_alphas" in name:
                gate_sq += norm * norm
            else:
                other_sq += norm * norm
        self._diag_gate_grad_sq += gate_sq
        self._diag_other_grad_sq += other_sq
        return summary

    def after_epoch(self):
        result = original_after_epoch(self)
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        learner = model.prompt_learner
        records.setdefault("gate_type", learner.gate_type)
        lr_groups = []
        for name, optim in self._optims.items():
            lr_groups.append(
                {
                    "model": name,
                    "lrs": [float(group["lr"]) for group in optim.param_groups],
                }
            )
        records["epochs"].append(
            {
                "epoch": self.epoch + 1,
                "gate_layers": gate_layer_summary(learner),
                "gate_grad_norm_epoch_l2": self._diag_gate_grad_sq ** 0.5,
                "other_grad_norm_epoch_l2": self._diag_other_grad_sq ** 0.5,
                "train_loss_mean": self._diag_loss_sum / max(self._diag_n_batches, 1),
                "lr_groups": lr_groups,
            }
        )
        return result

    MaPLe.before_epoch = before_epoch
    MaPLe.forward_backward = forward_backward
    MaPLe.after_epoch = after_epoch

    sys.argv = ["train.py", *args.train_args]
    try:
        runpy.run_path("train.py", run_name="__main__")
        records["status"] = "success"
    finally:
        MaPLe.before_epoch = original_before_epoch
        MaPLe.forward_backward = original_forward_backward
        MaPLe.after_epoch = original_after_epoch
        records["argv"] = sys.argv.copy()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(records, handle, indent=2)
        print("GATE_TRAJECTORY_REPORT", json.dumps(records))


if __name__ == "__main__":
    main()
