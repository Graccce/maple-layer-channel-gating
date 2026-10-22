"""Replay the complete pinned original trainer without modifying the fixture or normal registry.

Usage: python scripts/maple/run_upstream_reference.py METRICS_JSON [train.py args]
Runs only the none recipe. The sole source transformation removes registry
registration, allowing explicit temporary replacement of the existing trainer.
"""
import ast
import hashlib
from pathlib import Path
import runpy
import sys

FIXTURE_SHA256 = 'e067414c25e87a7ea40b9ec399fd13f655fe38a5f205e3bc5a5ce74a78a761b0'


def source_tree(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != FIXTURE_SHA256:
        raise ValueError('Pinned upstream source checksum mismatch')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MaPLe')
    if len(trainer.decorator_list) != 1:
        raise ValueError('Unexpected upstream trainer registration')
    trainer.decorator_list = []
    return tree


def validate_reference_family(state):
    if any(any(tag in key for tag in ('gate_alphas', '_gate_schema', '_frozen_projection_schema'))
           for key in state):
        raise ValueError('Pinned original reference cannot load gated or frozen-projection checkpoints')


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from dassl.engine import TRAINER_REGISTRY
    import trainers.maple as current
    fixture = root/'tests/fixtures/maple_upstream.py'
    namespace = {'__name__':'pinned_original_maple'}
    exec(compile(source_tree(fixture), str(fixture), 'exec'), namespace)
    reference = namespace['MaPLe']
    original_check = reference.check_cfg
    original_load = reference.load_model

    def check(self, cfg):
        original_check(self, cfg)
        if cfg.TRAINER.MAPLE.GATE_TYPE != 'none':
            raise ValueError('Pinned original reference requires GATE_TYPE=none')
        if cfg.RESUME or cfg.MODEL.INIT_WEIGHTS:
            raise ValueError('Reference requires fresh training or explicit matching-checkpoint evaluation')
    reference.check_cfg = check

    def load(self, directory, epoch=None):
        if directory:
            from dassl.utils import load_checkpoint
            filename = 'model-best.pth.tar' if epoch is None else f'model.pth.tar-{epoch}'
            for name in self.get_model_names():
                state = load_checkpoint(str(Path(directory)/name/filename))['state_dict']
                validate_reference_family(state)
        return original_load(self, directory, epoch)
    reference.load_model = load
    original_class = current.MaPLe
    original_registry = TRAINER_REGISTRY._obj_map['MaPLe']
    try:
        current.MaPLe = reference
        TRAINER_REGISTRY._obj_map['MaPLe'] = reference
        print('PINNED_ORIGINAL_REFERENCE', FIXTURE_SHA256)
        sys.argv = ['profile_run.py', *sys.argv[1:]]
        runpy.run_path('scripts/maple/profile_run.py',run_name='__main__')
    finally:
        current.MaPLe = original_class
        TRAINER_REGISTRY._obj_map['MaPLe'] = original_registry


if __name__ == '__main__':
    main()

