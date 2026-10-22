"""Opt-in frozen-deep-projection control around the existing profiled trainer.

Usage: python scripts/maple/run_projection_control.py METRICS_JSON AUDIT_JSON [train.py args]
The shallow projection remains trainable. Checkpoints have a strict extra schema
buffer and must be evaluated through this wrapper, with the matching gate mode.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import torch


def freeze_projections(model):
    learner = model.prompt_learner
    layers = learner.compound_prompt_projections
    if not len(layers):
        raise ValueError('Frozen projection control requires deep prompt projections')
    learner.register_buffer('_frozen_projection_schema', torch.tensor([1, len(layers)], dtype=torch.int64))
    for parameter in layers.parameters():
        parameter.requires_grad_(False)
    return {name: value.detach().cpu().clone() for name, value in layers.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    metrics_path, audit_path = map(Path, sys.argv[1:3])
    argv = sys.argv[3:]
    if metrics_path.exists() or audit_path.exists():
        raise FileExistsError('Refusing to overwrite control results')
    if '--resume' in argv:
        raise ValueError('This control only supports fresh training or explicit evaluation')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no experiment started')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import trainers.maple as maple
    original_optimizer = maple.build_optimizer
    captured = []
    facts = dict(status='failed', control='freeze_deep_prompt_projections',
                 shallow_projection='trainable', evaluation_only='--eval-only' in argv)

    def optimizer(model, cfg):
        initial = freeze_projections(model)
        if cfg.STAGED_LR:
            raise ValueError('This control requires the standard non-staged optimizer')
        result = original_optimizer(model, cfg,
            param_groups=[{'params': [p for p in model.parameters() if p.requires_grad]}])
        frozen = set(map(id, model.prompt_learner.compound_prompt_projections.parameters()))
        if any(id(p) in frozen for group in result.param_groups for p in group['params']):
            raise RuntimeError('Optimizer unexpectedly contains frozen projections')
        captured.append((model, initial))
        facts['initial_projection_sha256'] = state_digest(initial)
        facts['gate_type'] = model.prompt_learner.gate_type
        facts['projection_count'] = len(model.prompt_learner.compound_prompt_projections)
        facts['actual_trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('FROZEN_PROJECTION_CONTROL', json.dumps(facts))
        return result

    maple.build_optimizer = optimizer
    try:
        sys.argv = ['profile_run.py', str(metrics_path), *argv]
        runpy.run_path('scripts/maple/profile_run.py', run_name='__main__')
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one model')
        model, initial = captured[0]
        final = model.prompt_learner.compound_prompt_projections.state_dict()
        unchanged = all(torch.equal(initial[name], value.cpu()) for name, value in final.items())
        facts['final_projection_sha256'] = state_digest(final)
        facts['projection_parameters_frozen'] = all(not p.requires_grad and p.grad is None
            for p in model.prompt_learner.compound_prompt_projections.parameters())
        facts['projection_unchanged_during_training'] = unchanged if not facts['evaluation_only'] else None
        # Evaluation loads trained projections; training must preserve initialization exactly.
        if not facts['projection_parameters_frozen'] or (not facts['evaluation_only'] and not unchanged):
            raise RuntimeError('Frozen projection invariant failed')
        facts['status'] = 'success'
    finally:
        maple.build_optimizer = original_optimizer
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open('x') as handle:
            json.dump(facts, handle, indent=2)
        print('PROJECTION_CONTROL_REPORT', json.dumps(facts))


if __name__ == '__main__':
    main()

