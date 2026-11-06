"""Paper figures from committed result files (no GPU, no checkpoints).

Usage: python scripts/maple/plot_paper_figures.py
Reads docs/results/2026-11-06-v2-summary.json and docs/results/gate_stats/*_new.json,
writes docs/figures/fig_aircraft_stability.{png,pdf} and fig_gate_deviation.{png,pdf}.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'docs/results'
OUT = ROOT / 'docs/figures'

# Reference categorical palette (light), validated: slots 1-4 pass adjacent CVD/normal-vision checks.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
TEXT, TEXT_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COLLAPSE_HM = 20.0

plt.rcParams.update({
    'font.family': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'axes.titlecolor': TEXT,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def aircraft_stability():
    table = json.loads((RESULTS / '2026-11-06-v2-summary.json').read_text())['all_seeds']
    methods = [('none', '原版 MaPLe'), ('channel', '静态 channel\n门控'),
               ('none_clip', '原版\n+ 梯度裁剪'), ('channel_clip', '静态 channel\n+ 梯度裁剪')]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for x, (tag, _) in enumerate(methods):
        per_seed = table[f'fgvc_aircraft/{tag}']['per_seed_hm']
        seeds = sorted(per_seed, key=int)
        values = [per_seed[s] for s in seeds]
        offsets = [(i - 2) * 0.09 for i in range(len(values))]
        for dx, s, v in zip(offsets, seeds, values):
            collapsed = v < COLLAPSE_HM
            ax.scatter(x + dx, v, s=60 if collapsed else 46, color=SERIES[1] if collapsed else SERIES[0],
                       marker='X' if collapsed else 'o', edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if collapsed:
                ax.annotate(f's{s}', (x + dx, v), xytext=(6, -3), textcoords='offset points',
                            fontsize=8, color=TEXT_2)
        mean = sum(values) / len(values)
        ax.hlines(mean, x - 0.28, x + 0.28, color=TEXT, linewidth=2, zorder=2)
        ax.annotate(f'均值 {mean:.1f}', (x + 0.3, mean), xytext=(0, 0), textcoords='offset points',
                    fontsize=8, color=TEXT_2, va='center')
    ax.set_xticks(range(len(methods)), [label for _, label in methods])
    ax.set_xlim(-0.5, len(methods) - 0.2)
    ax.set_ylim(10, 42)
    ax.set_ylabel('HM (%)')
    ax.set_title('FGVCAircraft 五个种子的 HM：原版也会崩溃，梯度裁剪不能消除', loc='left', fontsize=11)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=SERIES[0], markersize=7, label='正常'),
               plt.Line2D([], [], marker='X', linestyle='', color=SERIES[1], markersize=7,
                          label=f'崩溃（HM < {COLLAPSE_HM:.0f}，标注种子号）'),
               plt.Line2D([], [], color=TEXT, linewidth=2, label='五种子均值')]
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=8, ncol=3)
    save(fig, 'fig_aircraft_stability')


def gate_deviation():
    datasets = [('dtd', 'DTD'), ('eurosat', 'EuroSAT'), ('oxford_flowers', 'Flowers'),
                ('fgvc_aircraft', 'Aircraft')]
    markers = ['o', 's', '^', 'D']
    # End-label offsets (points) so the four direct labels at layer 8 do not collide.
    label_dy = {'dtd': 7, 'oxford_flowers': 0, 'fgvc_aircraft': -6, 'eurosat': -4}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.grid(axis='y', color=GRID, linewidth=0.8, which='major')
    ax.set_axisbelow(True)
    for (key, label), color, marker in zip(datasets, SERIES, markers):
        layers = json.loads((RESULTS / f'gate_stats/{key}_new.json').read_text())['layers']
        xs = [layer['layer'] for layer in layers]
        ys = [layer['mean_abs_dev'] for layer in layers]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(10, label_dy[key]), textcoords='offset points',
                    fontsize=8, color=TEXT_2, va='center')
    ax.axhline(0.5, color=TEXT_2, linewidth=1, linestyle='--')
    ax.annotate('门控允许的最大偏离 0.5（g ∈ (0.5, 1.5)）', (1, 0.5), xytext=(0, -12),
                textcoords='offset points', fontsize=8, color=TEXT_2)
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(0.7, 9.0)
    ax.set_xticks(range(1, 9))
    ax.set_xlabel('带深层 prompt 的视觉层序号 l')
    ax.set_ylabel('平均 |g_l(x) − 1|（对数刻度）')
    ax.set_title('条件门控训练后几乎停在恒等映射（Novel 测试集，seed 1）', loc='left', fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.86), frameon=False, fontsize=8, ncol=4)
    save(fig, 'fig_gate_deviation')


if __name__ == '__main__':
    aircraft_stability()
    gate_deviation()
    print('written to', OUT)
