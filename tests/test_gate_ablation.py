import importlib.util
from pathlib import Path
import types
import unittest
import torch

spec = importlib.util.spec_from_file_location('ablation', Path(__file__).resolve().parents[1] /
                                            'scripts/maple/run_gate_ablation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GateAblationTests(unittest.TestCase):
    def test_fold_preserves_original_layers_and_handles_scalar_and_channel(self):
        for alpha in (torch.tensor(.2), torch.tensor([.2, -.3, .1])):
            layers = torch.nn.ModuleList([torch.nn.Linear(2, 3), torch.nn.Linear(2, 3)])
            learner = types.SimpleNamespace(compound_prompt_projections=layers,
                                            gate_alphas=[alpha, alpha])
            saved = {k: v.clone() for k, v in layers.state_dict().items()}
            folded = module.folded_projections(learner)
            x = torch.randn(4, 2)
            gate = 1 + .5 * alpha.tanh()
            for original, result in zip(layers, folded):
                torch.testing.assert_close(original(x)*gate, result(x), atol=1e-6, rtol=1e-6)
            for k, v in layers.state_dict().items():
                self.assertTrue(torch.equal(v, saved[k]))


if __name__ == '__main__':
    unittest.main()

