"""Opt-in diagnostics around the existing train.py; never converts saved checkpoints."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import sys
import torch


def folded_projections(learner):
    layers = copy.deepcopy(learner.compound_prompt_projections)
    with torch.no_grad():
        for layer, alpha in zip(layers, learner.gate_alphas):
            gate = (1 + .5 * alpha.float().tanh()).to(layer.weight.dtype)
            layer.weight.mul_(gate.reshape(-1, 1))
            layer.bias.mul_(gate)
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['fixed_one', 'fold_check'])
    parser.add_argument('report', type=Path)
    parser.add_argument('train_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f'Refusing to overwrite {args.report}')
    if not torch.cuda.is_available():
        raise SystemExit('Requires CUDA; no experiment started')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from trainers.maple import MaPLe
    facts = dict(kind=args.kind, status='failed', evaluated_examples=0,
                 prediction_disagreements=0, max_abs_logit_error=0.0,
                 original_correct=0, folded_correct=0)
    learners = []
    old_build = MaPLe.build_model
    old_load = MaPLe.load_model

    def build(self):
        old_build(self)
        if isinstance(self.model, torch.nn.DataParallel):
            raise RuntimeError('Diagnostic supports one GPU only')
        learner = self.model.prompt_learner
        learners.append(learner)
        if learner.gate_type == 'none':
            raise ValueError('Diagnostic requires scalar or channel')
        facts['gate_type'] = learner.gate_type
        if args.kind == 'fixed_one':
            with torch.no_grad():
                for alpha in learner.gate_alphas:
                    alpha.zero_()
                    alpha.requires_grad_(False)
            facts['gate_training'] = 'frozen zero alpha; effective gate exactly one'

    def load(self, *a, **kw):
        old_load(self, *a, **kw)
        if args.kind != 'fold_check':
            return
        model = self.model
        learner = model.prompt_learner
        folded = folded_projections(learner)
        original_forward = model.forward
        original_parse = self.parse_batch_test
        batch_labels = []

        def parse(batch):
            image, label = original_parse(batch)
            batch_labels[:] = [label]
            return image, label
        self.parse_batch_test = parse

        def compare(image, label=None):
            if model.training:
                raise RuntimeError('Folding diagnostic is evaluation only')
            with torch.no_grad():
                original = original_forward(image, label)
                layers, mode = learner.compound_prompt_projections, learner.gate_type
                try:
                    learner.compound_prompt_projections = folded
                    learner.gate_type = 'none'
                    result = original_forward(image, label)
                finally:
                    learner.compound_prompt_projections = layers
                    learner.gate_type = mode
                assert torch.isfinite(original).all() and torch.isfinite(result).all()
                facts['evaluated_examples'] += image.shape[0]
                assert len(batch_labels) == 1 and batch_labels[0].shape[0] == image.shape[0]
                truth = batch_labels[0]
                facts['original_correct'] += (original.argmax(1) == truth).sum().item()
                facts['folded_correct'] += (result.argmax(1) == truth).sum().item()
                facts['prediction_disagreements'] += (original.argmax(1) != result.argmax(1)).sum().item()
                facts['max_abs_logit_error'] = max(facts['max_abs_logit_error'],
                                                 (original-result).abs().max().item())
            return original
        model.forward = compare
        facts['fold_scope'] = 'full CLIP on every evaluation batch; in-memory projection weights only'
        facts['fold_precision'] = 'projection native dtype; CLIP recipe fp16'

    MaPLe.build_model = build
    MaPLe.load_model = load
    argv = args.train_args
    if argv and argv[0] == '--':
        argv = argv[1:]
    if args.kind == 'fold_check' and '--eval-only' not in argv:
        raise SystemExit('fold_check requires --eval-only')
    if args.kind == 'fixed_one' and any(x in argv for x in ('--resume', '--model-dir', '--eval-only')):
        raise SystemExit('fixed_one is a fresh training diagnostic, not checkpoint evaluation/resume')
    try:
        sys.argv = ['train.py', *argv]
        runpy.run_path('train.py', run_name='__main__')
        if args.kind == 'fold_check':
            assert facts['evaluated_examples'] > 0
        else:
            assert learners and all(not p.requires_grad and torch.count_nonzero(p).item() == 0
                                    for p in learners[0].gate_alphas)
            facts['gates_verified_frozen_zero_after_training'] = True
        facts['status'] = 'success'
    finally:
        MaPLe.build_model = old_build
        MaPLe.load_model = old_load
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as f:
            json.dump(facts, f, indent=2)
        print('ABLATION_REPORT', json.dumps(facts))


if __name__ == '__main__':
    main()

