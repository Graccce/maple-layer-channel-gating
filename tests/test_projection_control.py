import importlib.util
from pathlib import Path
import unittest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('projection_control', Path(__file__).resolve().parents[1] /
    'scripts/maple/run_projection_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class ProjectionControlTests(unittest.TestCase):
    def test_freeze_preserves_prompts_and_gate_learning_and_separates_checkpoints(self):
        for width in (None, 1, 3):
            with self.subTest(gate_width=width):
                model = nn.Module()
                model.prompt_learner = learner = nn.Module()
                learner.compound_prompt_projections = nn.ModuleList([nn.Linear(2, 3)])
                learner.proj = nn.Linear(2, 3)
                learner.prompt = nn.Parameter(torch.randn(2, 2))
                if width is not None:
                    learner.alpha = nn.Parameter(torch.zeros(()) if width == 1 else torch.zeros(width))
                original = {k: v.clone() for k, v in model.state_dict().items()}
                initial = control.freeze_projections(model)
                optim = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
                projected = learner.compound_prompt_projections[0](learner.prompt)
                loss = (projected * (1+.5*learner.alpha.tanh()) if width else projected).square().sum()
                loss.backward()
                self.assertGreater(learner.prompt.grad.abs().sum().item(), 0)
                if width:
                    self.assertGreater(learner.alpha.grad.abs().sum().item(), 0)
                optim.step()
                self.assertTrue(learner.proj.weight.requires_grad)
                self.assertEqual(control.state_digest(initial),
                    control.state_digest(learner.compound_prompt_projections.state_dict()))
                self.assertTrue(all(p.grad is None for p in learner.compound_prompt_projections.parameters()))
                # Ordinary strict loading must reject the diagnostic-only schema.
                with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                    model.load_state_dict(original, strict=True)


if __name__ == '__main__':
    unittest.main()

