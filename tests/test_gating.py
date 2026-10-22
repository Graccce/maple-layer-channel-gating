"""CPU tests of real prompt learner; tiny CLIP embedding fixture, no weights/data downloads."""
import copy
import hashlib
import io
import subprocess
import tempfile
from pathlib import Path
import os
import types
import unittest
from unittest.mock import patch
import torch
from torch import nn
from dassl.config import get_cfg_default
from dassl.optim import build_optimizer
from trainers.maple import MultiModalPromptLearner, MaPLe
from trainers.gating_checkpoint import load_prompt_checkpoint
from train import extend_cfg

torch.set_num_threads(1)


class Embedding(nn.Module):
    def forward(self, tokens):
        return torch.sin(tokens.float().unsqueeze(-1) + torch.arange(512).float() * .01)


class TinyCLIP(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.ln_final = nn.LayerNorm(512)
        self.visual = types.SimpleNamespace(input_resolution=224)
        self.token_embedding = Embedding()


def config(mode='none', depth=3):
    c = get_cfg_default()
    extend_cfg(c)
    c.INPUT.SIZE = (224, 224)
    c.TRAINER.MAPLE.GATE_TYPE = mode
    c.TRAINER.MAPLE.PROMPT_DEPTH = depth
    c.TRAINER.MAPLE.PREC = 'fp32'
    c.MODEL.BACKBONE.NAME = 'ViT-B/16'
    return c


def learner(mode='none', depth=3):
    torch.manual_seed(13)
    m = MultiModalPromptLearner(config(mode, depth), ['cat', 'dog'], TinyCLIP())
    # Upstream unconditionally makes shallow projection fp16 even for fp32 CLIP.
    # Normalize fixture projection only; production precision is unchanged.
    m.proj.float()
    return m


class Wrapped(nn.Module):
    def __init__(self, mode='none', depth=3):
        super().__init__()
        self.prompt_learner = learner(mode, depth)


class GateTests(unittest.TestCase):
    def test_modes_shapes_identity_and_rng(self):
        base = learner()
        state = copy.deepcopy(base.state_dict())
        base_outputs = base()
        for mode in ['none', 'scalar', 'channel']:
            m = learner(mode)
            for k, v in state.items():
                self.assertTrue(torch.equal(v, m.state_dict()[k]), k)
            for index, output in enumerate(m()[3]):
                self.assertEqual(output.shape, (2, 768))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.equal(base_outputs[3][index], output))
            if mode != 'none':
                self.assertEqual(len(m.gate_alphas), len(m.compound_prompt_projections))
                self.assertEqual(m.gate_alphas[0].shape, torch.Size([] if mode == 'scalar' else [768]))
        self.assertFalse(any('gate' in k for k in base.state_dict()))

    def test_original_head_forward_exact(self):
        import ast
        fixture = Path(__file__).parent / 'fixtures/maple_upstream.py'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         'e067414c25e87a7ea40b9ec399fd13f655fe38a5f205e3bc5a5ce74a78a761b0')
        source = fixture.read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiModalPromptLearner')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        ns = {}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), '<HEAD-forward>', 'exec'), ns)
        m = learner()
        original = ns['forward'](m)
        current = m()
        for a, b in zip(original[:2], current[:2]):
            self.assertTrue(torch.equal(a, b))
        for a, b in zip(original[3], current[3]):
            self.assertTrue(torch.equal(a, b))

    def test_depth_and_invalid(self):
        for depth in [1, 2, 9]:
            for mode in ['none', 'scalar', 'channel']:
                m = learner(mode, depth)
                self.assertEqual(len(m()[3]), depth - 1)
        with self.assertRaisesRegex(ValueError, 'GATE_TYPE'):
            learner('invalid')

    def test_gradients_and_actual_dassl_optimizer(self):
        for mode in ['scalar', 'channel']:
            m = learner(mode)
            opt = build_optimizer(m, config(mode).OPTIM)
            before = [p.detach().clone() for p in m.gate_alphas]
            loss = sum(z.square().mean() for z in m()[3])
            loss.backward()
            for p in m.gate_alphas:
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)
            opt.step()
            self.assertTrue(all(not torch.equal(a, b) for a, b in zip(before, m.gate_alphas)))

    def test_broadcast_and_dtype(self):
        m = learner('channel')
        with torch.no_grad():
            m.gate_alphas[0].copy_(torch.linspace(-2, 2, 768))
        z = m.compound_prompt_projections[0](m.compound_prompts_text[0])
        torch.testing.assert_close(m()[3][0], z * (1 + .5 * m.gate_alphas[0].tanh()))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            outputs = m()[3]
        self.assertTrue(all(z.dtype == torch.bfloat16 and torch.isfinite(z).all() for z in outputs))

    def test_checkpoint_roundtrip_and_mismatch(self):
        for source in ['none', 'scalar', 'channel']:
            saved = Wrapped(source).state_dict()
            f = io.BytesIO()
            torch.save({'state_dict': saved}, f)
            f.seek(0)
            saved = torch.load(f, map_location='cpu')['state_dict']
            for target in ['none', 'scalar', 'channel']:
                dst = Wrapped(target)
                if source == target:
                    load_prompt_checkpoint(dst, saved)
                    dst.load_state_dict(saved)  # Dassl strict resume route
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Gate'):
                        load_prompt_checkpoint(dst, saved)
                    with self.assertRaises(RuntimeError):
                        dst.load_state_dict(saved, strict=False)
        saved = Wrapped('channel').state_dict()
        saved.pop('prompt_learner.gate_alphas.0')
        with self.assertRaisesRegex(RuntimeError, 'count/shape'):
            load_prompt_checkpoint(Wrapped('channel'), saved)
        legacy = Wrapped().state_dict()
        legacy['prompt_learner.gate_alphas'] = torch.zeros(2, 768)
        with self.assertRaisesRegex(RuntimeError, 'Legacy gate'):
            load_prompt_checkpoint(Wrapped('channel'), legacy)

    def test_strict_learned_keys_and_target_tokens(self):
        m = Wrapped()
        state = m.state_dict()
        state['prompt_learner.token_prefix'] = torch.zeros(99, 1, 512)
        load_prompt_checkpoint(m, state)
        state.pop('prompt_learner.ctx')
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            load_prompt_checkpoint(m, state)
        with self.assertRaises(RuntimeError):
            load_prompt_checkpoint(Wrapped('scalar', 4), Wrapped('scalar', 3).state_dict())

    def test_config_files_and_cli_merge(self):
        for mode in ['none', 'scalar', 'channel']:
            c = config()
            c.merge_from_file(f'configs/trainers/MaPLe/vit_b16_gating_{mode}.yaml')
            self.assertEqual(c.TRAINER.MAPLE.GATE_TYPE, mode)
            c.merge_from_list(['TRAINER.MAPLE.GATE_TYPE', mode])
        from train import setup_cfg
        import argparse
        args = argparse.Namespace(root='', output_dir='', resume='', seed=1,
                                  source_domains=None, target_domains=None, transforms=None,
                                  trainer='MaPLe', backbone='', head='',
                                  dataset_config_file='configs/datasets/oxford_pets.yaml',
                                  config_file='configs/trainers/MaPLe/vit_b16_gating_channel.yaml',
                                  opts=['TRAINER.MAPLE.GATE_TYPE', 'scalar'])
        self.assertEqual(setup_cfg(args).TRAINER.MAPLE.GATE_TYPE, 'scalar')

    def test_trainer_freezing_and_optimizer_membership(self):
        class Model(nn.Module):
            def __init__(self, cfg, names, clip):
                super().__init__()
                self.prompt_learner = learner(cfg.TRAINER.MAPLE.GATE_TYPE)
                self.image_encoder = nn.Linear(2, 2)
                self.text_encoder = nn.Linear(2, 2)
        t = MaPLe.__new__(MaPLe)
        t.cfg = config('channel')
        t.dm = types.SimpleNamespace(dataset=types.SimpleNamespace(classnames=['cat', 'dog']))
        t.device = torch.device('cpu')
        t.register_model = lambda *args: None
        with patch('trainers.maple.load_clip_to_cpu', return_value=nn.Linear(1, 1)), \
             patch('trainers.maple.CustomCLIP', Model):
            t.build_model()
        self.assertTrue(all(not p.requires_grad for p in t.model.image_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in t.model.text_encoder.parameters()))
        optimized = {id(p) for g in t.optim.param_groups for p in g['params']}
        self.assertTrue(all(id(p) in optimized for p in t.model.prompt_learner.gate_alphas))

    def test_experiment_script_modes_and_overwrite_guard(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts/maple').mkdir(parents=True)
            (root / 'data/oxford_pets').mkdir(parents=True)
            (root / 'data/oxford_pets/split_zhou_OxfordPets.json').write_text('{}')
            shutil.copy('scripts/maple/run_gating.sh', root / 'scripts/maple/run_gating.sh')
            tools = root / 'bin'
            tools.mkdir()
            for name, body in {
                'python': '#!/bin/sh\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n',
                'git': '#!/bin/sh\nexit 0\n',
                'tar': '#!/bin/sh\nexit 0\n',
            }.items():
                p = tools / name
                p.write_text(body)
                p.chmod(0o755)
            env = dict(os.environ, PATH=str(tools) + ':' + os.environ['PATH'], CALL_LOG=str(root / 'calls'))
            for mode in ['none', 'scalar', 'channel']:
                cmd = ['bash', str(root / 'scripts/maple/run_gating.sh'), str(root / 'data'),
                       'oxford_pets', mode, '1', 'script_test', 'smoke']
                subprocess.run(cmd, env=env, check=True, capture_output=True)
                calls = (root / 'calls').read_text()
                self.assertIn(f'vit_b16_gating_{mode}.yaml', calls)
                self.assertIn('DATASET.SUBSAMPLE_CLASSES base', calls)
                self.assertIn('DATASET.SUBSAMPLE_CLASSES new', calls)
                self.assertNotEqual(subprocess.run(cmd, env=env, capture_output=True).returncode, 0)
            # Dataset aliases and opt-in control must route through the same runner.
            (root / 'data/caltech-101').mkdir(parents=True)
            command = ['bash', str(root / 'scripts/maple/run_gating.sh'), str(root / 'data'),
                       'caltech101', 'channel', '1', 'frozen_test', 'formal', 'frozen_projection']
            missing = subprocess.run(command, env=env, capture_output=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn(b'Missing fixed split', missing.stdout)
            (root / 'data/caltech-101/split_zhou_Caltech101.json').write_text('{}')
            subprocess.run(command, env=env, check=True, capture_output=True)
            self.assertIn('run_projection_control.py', (root / 'calls').read_text())
            self.assertTrue((root / 'output/gating_v1/ablation/formal/frozen_test/caltech101/channel/seed1').is_dir())

    def test_result_summary_formula_and_refusal(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'formal/run/oxford_pets/channel/seed1'
            for subset, accuracy in [('base', 80), ('new', 60)]:
                p = run / f'test_{subset}'
                p.mkdir(parents=True)
                (p / 'log.txt').write_text(f'* accuracy: {accuracy}%\n')
            (run / 'train_metrics.json').write_text(json.dumps(dict(status='success', trainable_parameters=100,
                gate_parameters=10, trainer_train_seconds_including_final_eval=1,
                peak_allocated_bytes=1024, peak_reserved_bytes=2048)))
            subprocess.run([os.sys.executable, 'scripts/maple/summarize_gating.py', str(run)],
                           check=True, capture_output=True)
            row = json.loads((run / 'results.json').read_text())
            self.assertAlmostEqual(row['harmonic_mean'], 2*80*60/140)
            (run / 'test_new/log.txt').write_text('* accuracy: 70%\n')
            self.assertNotEqual(subprocess.run([os.sys.executable, 'scripts/maple/summarize_gating.py', str(run)],
                                               capture_output=True).returncode, 0)

    def test_real_dassl_save_resume_and_maple_eval_loader(self):
        from dassl.utils import save_checkpoint, resume_from_checkpoint
        for mode in ['none', 'scalar', 'channel']:
            with tempfile.TemporaryDirectory() as tmp:
                src = Wrapped(mode)
                opt = build_optimizer(src, config(mode).OPTIM)
                sum(z.square().mean() for z in src.prompt_learner()[3]).backward()
                opt.step()
                directory = Path(tmp) / 'MultiModalPromptLearner'
                save_checkpoint(dict(state_dict=src.state_dict(), optimizer=opt.state_dict(), epoch=5), str(directory))
                dst = Wrapped(mode)
                dst_opt = build_optimizer(dst, config(mode).OPTIM)
                self.assertEqual(resume_from_checkpoint(str(directory), dst, dst_opt), 5)
                for k, value in src.state_dict().items():
                    self.assertTrue(torch.equal(value, dst.state_dict()[k]))
                t = MaPLe.__new__(MaPLe)
                t._models = {'MultiModalPromptLearner': Wrapped(mode)}
                t.get_model_names = lambda: ['MultiModalPromptLearner']
                t.load_model(tmp, epoch=5)
                if mode != 'none':
                    with self.assertRaises(RuntimeError):
                        resume_from_checkpoint(str(directory), Wrapped())


if __name__ == '__main__':
    unittest.main(verbosity=2)

