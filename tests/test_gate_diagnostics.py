"""CPU-only checks for the diagnostic wrappers' pure logic."""

import unittest

import torch

from scripts.maple.run_gate_lr_ablation import build_gate_lr_param_groups
from scripts.maple.run_gate_trajectory import gate_layer_summary


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.gate_alphas = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(()))]
        )
        self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)


class _Cfg:
    LR = 0.0035


class TestGateLrParamGroups(unittest.TestCase):
    def test_gate_group_gets_scaled_lr(self):
        model = _Tiny()
        groups = build_gate_lr_param_groups(model, _Cfg(), 0.1)
        self.assertEqual(len(groups), 2)
        other, gate = groups
        self.assertEqual(other["lr"], 0.0035)
        self.assertAlmostEqual(gate["lr"], 0.00035)
        self.assertEqual(len(gate["params"]), 2)
        self.assertEqual(len(other["params"]), 2)

    def test_frozen_params_excluded(self):
        model = _Tiny()
        groups = build_gate_lr_param_groups(model, _Cfg(), 10.0)
        all_params = [p for group in groups for p in group["params"]]
        self.assertTrue(all(p is not model.frozen for p in all_params))


class TestGateLayerSummary(unittest.TestCase):
    def test_identity_initialization(self):
        model = _Tiny()
        rows = gate_layer_summary(model)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row["gate_mean"], 1.0, places=6)
            self.assertAlmostEqual(row["max_abs_deviation_from_one"], 0.0, places=6)

    def test_scalar_and_channel_numel(self):
        model = _Tiny()
        rows = gate_layer_summary(model)
        self.assertEqual(rows[0]["numel"], 4)
        self.assertEqual(rows[1]["numel"], 1)


if __name__ == "__main__":
    unittest.main()
