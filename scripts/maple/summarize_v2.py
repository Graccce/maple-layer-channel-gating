"""Aggregate phase-2 runs with phase-1 paired baselines.

Usage: python scripts/maple/summarize_v2.py OUT_JSON ROOT [ROOT ...]
Each ROOT is searched for */DATASET/TAG/seedN/results.json (phase-1 gating_v1 and phase-2 v2 layouts
are identical below the run id). The same (dataset, tag, seed) appearing twice is an error, so a rerun
can never silently replace a result. HM is computed per seed, then mean +- sample std; dHM is the paired
difference to `none` on the same dataset and seed (seeds without a `none` partner are not paired).
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DATASETS = ['oxford_pets', 'dtd', 'caltech101', 'oxford_flowers', 'eurosat', 'fgvc_aircraft']


def collect(roots):
    runs = {}
    for root in roots:
        for path in Path(root).glob('**/seed*/results.json'):
            seed_dir = path.parent
            key = (seed_dir.parent.parent.name, seed_dir.parent.name, int(seed_dir.name[4:]))
            if key[0] not in DATASETS:
                continue
            if key in runs:
                raise RuntimeError(f'Duplicate result for {key}: {runs[key]["run"]} and {seed_dir}')
            row = json.loads(path.read_text())
            row['run'] = str(seed_dir)
            runs[key] = row
    return runs


def stats(values):
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'n': len(values)}


def summarize(runs, seeds=None):
    grouped = defaultdict(dict)
    for (dataset, tag, seed), row in runs.items():
        if seeds is None or seed in seeds:
            grouped[(dataset, tag)][seed] = row
    table = {}
    for (dataset, tag), by_seed in sorted(grouped.items()):
        base = grouped.get((dataset, 'none'), {})
        paired = [by_seed[s]['harmonic_mean'] - base[s]['harmonic_mean'] for s in by_seed if s in base]
        table[f'{dataset}/{tag}'] = {
            'seeds': sorted(by_seed),
            'base': stats([r['base_accuracy'] for r in by_seed.values()]),
            'novel': stats([r['novel_accuracy'] for r in by_seed.values()]),
            'hm': stats([r['harmonic_mean'] for r in by_seed.values()]),
            'delta_hm_vs_none': stats(paired) if paired else None,
            'per_seed_hm': {s: by_seed[s]['harmonic_mean'] for s in sorted(by_seed)},
            'trainable_parameters': next(iter(by_seed.values())).get('trainable_parameters'),
        }
    return table


def markdown(table, tags):
    lines = ['| 数据集 | 方法 | Base | Novel | HM | ΔHM vs none |', '|---|---|---:|---:|---:|---:|']
    averages = defaultdict(lambda: defaultdict(list))
    for dataset in DATASETS:
        for tag in tags:
            cell = table.get(f'{dataset}/{tag}')
            if not cell:
                continue
            d = cell['delta_hm_vs_none']
            delta = '—' if tag == 'none' or d is None else '{:+.2f}'.format(d['mean'])
            lines.append(f"| {dataset} | {tag} | {cell['base']['mean']:.2f} | {cell['novel']['mean']:.2f} | "
                         f"{cell['hm']['mean']:.2f} ± {cell['hm']['std']:.2f} | "
                         f"{delta} |")
            for k in ('base', 'novel', 'hm'):
                averages[tag][k].append(cell[k]['mean'])
    for tag in tags:
        a = averages[tag]
        if len(a['hm']) == len(DATASETS):
            lines.append(f"| **平均** | {tag} | {statistics.mean(a['base']):.2f} | {statistics.mean(a['novel']):.2f} | "
                         f"{statistics.mean(a['hm']):.2f} | |")
    return '\n'.join(lines)


if __name__ == '__main__':
    out, roots = sys.argv[1], sys.argv[2:]
    runs = collect(roots)
    main = summarize(runs, seeds={1, 2, 3})
    all_seeds = summarize(runs)
    Path(out).write_text(json.dumps({'seeds_1_3': main, 'all_seeds': all_seeds}, indent=2))
    tags = ['none', 'scalar', 'channel', 'cond_channel', 'kc', 'cond_channel_kc']
    print(markdown(main, tags))
