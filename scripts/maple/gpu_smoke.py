"""One real CLIP batch, no dataset and no saved training results. Requires GPU mode."""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import types
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train import extend_cfg
from dassl.config import get_cfg_default
from trainers.maple import MaPLe

if not torch.cuda.is_available():
    raise SystemExit('Requires AutoDL GPU mode; no model loaded.')
mode = sys.argv[1] if len(sys.argv) == 2 else 'channel'
cfg = get_cfg_default()
extend_cfg(cfg)
cfg.merge_from_file(f'configs/trainers/MaPLe/vit_b16_gating_{mode}.yaml')
torch.manual_seed(1)
t = MaPLe.__new__(MaPLe)
t.cfg = cfg
t.dm = types.SimpleNamespace(dataset=types.SimpleNamespace(classnames=['cat', 'dog']))
t.device = torch.device('cuda')
t.register_model = lambda *a: None
t.build_model()
m = t.model
image = torch.randn(2, 3, 224, 224, device='cuda')
label = torch.tensor([0, 1], device='cuda')
m.eval()
with torch.no_grad():
    logits = m(image)
    assert torch.isfinite(logits).all()
    # Pinned upstream fixture survives commits to this repository's HEAD.
    fixture = Path(__file__).resolve().parents[2] / 'tests/fixtures/maple_upstream.py'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == 'e067414c25e87a7ea40b9ec399fd13f655fe38a5f205e3bc5a5ce74a78a761b0'
    tree = ast.parse(fixture.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MultiModalPromptLearner')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<HEAD-forward>', 'exec'), ns)
    current = m.prompt_learner.forward
    try:
        m.prompt_learner.forward = types.MethodType(ns['forward'], m.prompt_learner)
        original = m(image)
    finally:
        m.prompt_learner.forward = current
    torch.testing.assert_close(logits, original, atol=0, rtol=0)
gates = list(getattr(m.prompt_learner, 'gate_alphas', []))
before = [p.detach().clone() for p in gates]
m.train()
loss = m(image, label)
assert torch.isfinite(loss)
t.optim.zero_grad()
loss.backward()
for p in gates:
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
assert all(p.grad is None for p in m.image_encoder.parameters() if not p.requires_grad)
assert all(p.grad is None for p in m.text_encoder.parameters() if not p.requires_grad)
t.optim.step()
assert all(not torch.equal(a, b) for a, b in zip(before, gates))
print('GPU_SMOKE_PASS', mode, 'loss', loss.item(), 'max_allocated', torch.cuda.max_memory_allocated())
