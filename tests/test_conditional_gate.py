"""CPU tests for the phase-2 input-conditioned gate, consistency loss config and grad clipping."""
import copy
import io
import unittest
import torch
from torch import nn
from clip.model import ResidualAttentionBlock_MaPLe
from trainers.maple import ConditionalGate
from tests.test_gating import learner, Wrapped, config
from trainers.gating_checkpoint import load_prompt_checkpoint

torch.set_num_threads(1)
COND = ['cond_channel', 'cond_scalar']


class Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def injected_tokens(deep_prompts, x):
    """Run one prompted visual block with attention/MLP zeroed: output == input after injection."""
    block = ResidualAttentionBlock_MaPLe(768, 12, design_details={'maple_length': 2}, i=1)
    block.attention = lambda t: torch.zeros_like(t)
    block.mlp = Zero()
    return block([x, deep_prompts, 0])[0]


class ConditionalGateTests(unittest.TestCase):
    def test_construction_keeps_rng_and_shared_init(self):
        torch.manual_seed(0)
        base = learner('none')
        after_none = torch.get_rng_state()
        state = base.state_dict()
        for mode in COND:
            m = learner(mode)
            self.assertTrue(torch.equal(after_none, torch.get_rng_state()), mode)
            for k, v in state.items():
                self.assertTrue(torch.equal(v, m.state_dict()[k]), k)
            self.assertEqual(len(m.cond_gates), len(m.compound_prompt_projections))
            self.assertEqual(m.cond_gates[0].up.out_features, 768 if mode == 'cond_channel' else 1)

    def test_identity_at_init_matches_none_bitwise(self):
        x = torch.randn(10, 3, 768).half()
        base_prompts = learner('none')()[3]
        for mode in COND:
            prompts = learner(mode)()[3]
            self.assertIsInstance(prompts[0], tuple)
            for (z, _), z0 in zip(prompts, base_prompts):
                self.assertTrue(torch.equal(z, z0))
            self.assertTrue(torch.equal(injected_tokens(prompts, x), injected_tokens(base_prompts, x)))

    def test_gate_is_input_dependent_after_update(self):
        gate = ConditionalGate(768, 8)
        with torch.no_grad():
            gate.up.weight.normal_(std=0.5)
        cls = torch.randn(4, 768)
        g = gate(cls)
        self.assertEqual(g.shape, (4, 768))
        self.assertTrue(((g > 0.5) & (g < 1.5)).all())
        # Different images get different gates -> not foldable into a static W.
        self.assertGreater((g[0] - g[1]).abs().max().item(), 1e-3)
        m = learner('cond_channel')
        with torch.no_grad():
            m.cond_gates[0].up.weight.normal_(std=0.5)
        x = torch.randn(10, 3, 768).half()
        out = injected_tokens(m()[3], x)
        z = m()[3][0][0]
        expected = (z.unsqueeze(0) * m.cond_gates[0](x[0]).unsqueeze(1)).permute(1, 0, 2).half()
        self.assertTrue(torch.equal(out[-2:], expected))
        self.assertTrue(torch.equal(out[:-2], x[:-2]))

    def test_gradients_reach_gate(self):
        m = learner('cond_channel')
        x = torch.randn(10, 3, 768)
        z, gate = m()[3][0]
        loss = (z.unsqueeze(0) * gate(x[0]).unsqueeze(1)).square().mean()
        loss.backward()
        self.assertGreater(m.cond_gates[0].up.weight.grad.abs().sum().item(), 0)

    def test_checkpoint_family_is_strict(self):
        modes = ['none', 'channel'] + COND
        for source in modes:
            f = io.BytesIO()
            torch.save({'state_dict': Wrapped(source).state_dict()}, f)
            f.seek(0)
            saved = torch.load(f, map_location='cpu')['state_dict']
            for target in modes:
                model = Wrapped(target)
                if source == target:
                    load_prompt_checkpoint(model, copy.deepcopy(saved))
                else:
                    with self.assertRaises(RuntimeError, msg=f'{source}->{target}'):
                        load_prompt_checkpoint(model, copy.deepcopy(saved))

    def test_new_options_default_to_original_recipe(self):
        c = config()
        m = c.TRAINER.MAPLE
        self.assertEqual((m.KC_TEXT_WEIGHT, m.KC_IMAGE_WEIGHT, m.KC_LOGIT_WEIGHT, m.GRAD_CLIP), (0, 0, 0, 0))
        self.assertEqual(m.GATE_RANK, 32)
        self.assertEqual(m.KC_REWEIGHT, 'none')


if __name__ == '__main__':
    unittest.main()
