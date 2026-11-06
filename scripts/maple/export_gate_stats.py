"""Export statistics of the input-conditioned gates g_l(x) on a test split.

Usage (same arguments as a train.py --eval-only call, plus an output JSON first):
  python scripts/maple/export_gate_stats.py OUT_JSON --root DATA --seed 1 --trainer MaPLe \
     --dataset-config-file configs/datasets/dtd.yaml --config-file configs/trainers/MaPLe/vit_b16_gating_none.yaml \
     --eval-only --model-dir RUN/train_base --load-epoch 5 DATASET.NUM_SHOTS 16 DATASET.SUBSAMPLE_CLASSES new \
     TRAINER.MAPLE.GATE_TYPE cond_channel

Per layer: mean/std over images of the channel-mean gate, mean |g-1|, fraction of (image, channel) pairs
with |g-1| > 0.05, mean per-image std across channels, the between-image std of g per channel
(averaged over channels: how input-dependent the gate is), and per-class mean gate deviation.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
out_path = Path(sys.argv.pop(1))
if out_path.exists():
    raise SystemExit(f'Refusing to overwrite {out_path}')

import runpy  # noqa: E402
from trainers.maple import MaPLe  # noqa: E402


def export(trainer):
    if trainer.cfg.TRAINER.MAPLE.GATE_TYPE not in ('cond_channel', 'cond_scalar'):
        raise SystemExit('Gate statistics need a cond_* GATE_TYPE')
    model = trainer.model.eval()
    gates = model.prompt_learner.cond_gates
    captured = {}
    hooks = [g.register_forward_hook(lambda m, i, o, idx=idx: captured.__setitem__(idx, o.detach().float().cpu()))
             for idx, g in enumerate(gates)]
    per_layer = defaultdict(list)
    labels = []
    correct = 0
    with torch.no_grad():
        for batch in trainer.test_loader:
            image, label = trainer.parse_batch_test(batch)
            logits = model(image)
            correct += (logits.argmax(1) == label).sum().item()
            labels.append(label.cpu())
            for idx in range(len(gates)):
                per_layer[idx].append(captured[idx])
    for h in hooks:
        h.remove()
    labels = torch.cat(labels)
    classnames = trainer.dm.dataset.classnames
    result = {'accuracy': 100.0 * correct / len(labels), 'num_images': len(labels), 'layers': []}
    for idx in range(len(gates)):
        g = torch.cat(per_layer[idx])  # [N, C]
        dev = (g - 1).abs()
        per_class = {}
        for c in labels.unique().tolist():
            per_class[classnames[c]] = dev[labels == c].mean().item()
        result['layers'].append({
            'layer': idx + 1,
            'image_mean_gate': {'mean': g.mean(1).mean().item(), 'std': g.mean(1).std().item()},
            'mean_abs_dev': dev.mean().item(),
            'max_abs_dev': dev.max().item(),
            'frac_dev_gt_0.05': (dev > 0.05).float().mean().item(),
            'within_image_channel_std': g.std(1).mean().item() if g.shape[1] > 1 else 0.0,
            'between_image_std': g.std(0).mean().item(),
            'per_class_mean_abs_dev': per_class,
            'channel_mean_gate': g.mean(0).tolist(),
        })
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'layers'}))
    for layer in result['layers']:
        print(layer['layer'], {k: round(v, 4) for k, v in layer.items() if isinstance(v, float)},
              'between_image_std', round(layer['between_image_std'], 4))


MaPLe.test = lambda self, split=None: export(self)
sys.argv = ['train.py', *sys.argv[1:]]
runpy.run_path('train.py', run_name='__main__')
