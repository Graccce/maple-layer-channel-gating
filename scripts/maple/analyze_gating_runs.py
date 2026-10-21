"""Summarize completed paired runs and inspect static gates without changing models."""
import argparse
import gc
import json
from pathlib import Path
import re
import statistics
import torch


def count_accuracy(path):
    text = path.read_text()
    totals = re.findall(r'\* total:\s*([\d,]+)', text)
    correct = re.findall(r'\* correct:\s*([\d,]+)', text)
    if len(totals) != 1 or len(correct) != 1:
        raise ValueError(f'Expected one completed evaluation: {path}')
    total, count = int(totals[0].replace(',', '')), int(correct[0].replace(',', ''))
    return dict(correct=count, total=total, accuracy=100 * count / total)


def summarize(root, epoch, inspect_gates=True):
    rows = []
    for mode in ('none', 'scalar', 'channel'):
        for run in sorted((root / mode).glob('seed*')):
            if not (run / 'results.json').exists():
                continue
            metrics = json.loads((run / 'train_metrics.json').read_text())
            assert metrics['status'] == 'success'
            base = count_accuracy(run / 'test_base/log.txt')
            novel = count_accuracy(run / 'test_new/log.txt')
            b, n = base['accuracy'], novel['accuracy']
            row = dict(mode=mode, seed=int(run.name[4:]), base=base, novel=novel,
                       harmonic_mean=2*b*n/(b+n), trainable_parameters=metrics['trainable_parameters'],
                       train_seconds_including_final_eval=metrics['trainer_train_seconds_including_final_eval'],
                       peak_allocated_mib=metrics['peak_allocated_bytes']/2**20,
                       peak_reserved_mib=metrics['peak_reserved_bytes']/2**20)
            if inspect_gates:
                checkpoint = torch.load(run / f'train_base/MultiModalPromptLearner/model.pth.tar-{epoch}',
                                        map_location='cpu')
                state = checkpoint['state_dict']
                gate_keys = sorted(k for k in state if k.startswith('prompt_learner.gate_alphas.'))
                projection_count = sum(k.startswith('prompt_learner.compound_prompt_projections.')
                                       and k.endswith('.weight') for k in state)
                assert len(gate_keys) == (0 if mode == 'none' else projection_count)
                layers = []
                for k in gate_keys:
                    index = k.rsplit('.', 1)[1]
                    alpha = state[k].double()
                    gate = 1 + .5*alpha.tanh()
                    w = state[f'prompt_learner.compound_prompt_projections.{index}.weight'].double()
                    bias = state[f'prompt_learner.compound_prompt_projections.{index}.bias'].double()
                    prompt = state[f'prompt_learner.compound_prompts_text.{index}'].double()
                    original = (prompt @ w.T + bias) * gate
                    folded = prompt @ (w * gate.reshape(-1, 1)).T + bias * gate
                    layers.append(dict(layer=int(index), gate_mean=gate.mean().item(),
                                       gate_std=gate.std(unbiased=False).item(),
                                       gate_min=gate.min().item(), gate_max=gate.max().item(),
                                       max_change_from_one=(gate-1).abs().max().item(),
                                       alpha_l2=alpha.norm().item(),
                                       fold_max_abs_error=(original-folded).abs().max().item()))
                row['gate_diagnostics'] = layers
                del checkpoint, state
                gc.collect()
            rows.append(row)
    groups = {}
    for mode in ('none', 'scalar', 'channel'):
        records = [r for r in rows if r['mode'] == mode]
        if not records:
            continue
        groups[mode] = dict(seeds=[r['seed'] for r in records], n=len(records))
        for key in ('harmonic_mean', 'train_seconds_including_final_eval', 'peak_allocated_mib'):
            values = [r[key] for r in records]
            groups[mode][key] = dict(mean=statistics.mean(values),
                                     sample_std=statistics.stdev(values) if len(values)>1 else None)
        for key in ('base', 'novel'):
            values = [r[key]['accuracy'] for r in records]
            groups[mode][key] = dict(mean=statistics.mean(values),
                                     sample_std=statistics.stdev(values) if len(values)>1 else None)
    paired = {}
    reference = {r['seed']:r for r in rows if r['mode']=='none'}
    for mode in ('scalar', 'channel'):
        differences = [r['harmonic_mean']-reference[r['seed']]['harmonic_mean']
                       for r in rows if r['mode']==mode and r['seed'] in reference]
        if differences:
            paired[mode] = dict(per_seed_hm_delta=differences, mean_hm_delta=statistics.mean(differences),
                                sample_std=statistics.stdev(differences) if len(differences)>1 else None)
    return dict(rows=rows, summary=groups, paired_comparison=paired,
                accuracy_source='exact correct/total counts; original results.json retains displayed rounded values')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_root, args.epoch)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report['summary'], indent=2))
