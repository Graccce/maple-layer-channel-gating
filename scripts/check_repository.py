"""Dependency-free checks for the publication's core invariants."""
import ast
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[1]
fixture = root / 'tests/fixtures/maple_upstream.py'
assert hashlib.sha256(fixture.read_bytes()).hexdigest() == 'e067414c25e87a7ea40b9ec399fd13f655fe38a5f205e3bc5a5ce74a78a761b0'
train = (root / 'train.py').read_text(encoding='utf-8')
assert 'cfg.TRAINER.MAPLE.GATE_TYPE = "none"' in train
for mode in ('none', 'scalar', 'channel'):
    path = root / f'configs/trainers/MaPLe/vit_b16_gating_{mode}.yaml'
    assert f'GATE_TYPE: "{mode}"' in path.read_text()
for path in [root/'train.py', *list((root/'trainers').glob('*.py')), *list((root/'scripts').rglob('*.py')), *list((root/'tests').rglob('*.py'))]:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path.relative_to(root)))
for path in [root/'tests/test_gating.py', root/'scripts/maple/gpu_smoke.py']:
    assert "['git', 'show', 'HEAD:trainers/maple.py']" not in path.read_text()
assert (root/'LICENSE').is_file()
assert (root/'UPSTREAM.md').is_file()
print('REPOSITORY_CHECKS_PASS')
