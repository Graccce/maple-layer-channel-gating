"""Measure a real existing train.py invocation; no separate training framework."""
import json
import os
from pathlib import Path
import runpy
import sys
import time
import torch

if len(sys.argv) < 3:
    raise SystemExit('Usage: python scripts/maple/profile_run.py METRICS_JSON [train.py arguments]')
if not torch.cuda.is_available():
    raise SystemExit('No CUDA device. Switch AutoDL to GPU mode before running experiments.')
metrics_path = Path(sys.argv[1])
if metrics_path.exists():
    raise SystemExit(f'Refusing to overwrite {metrics_path}')
# Import from the current worktree before train.py starts.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trainers.maple import MaPLe

metrics = {}
original_train = MaPLe.train


def measured_train(self, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        return original_train(self, *args, **kwargs)
    finally:
        torch.cuda.synchronize()
        metrics['trainer_train_seconds_including_final_eval'] = time.perf_counter() - start
        metrics['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        metrics['gate_parameters'] = sum(p.numel() for n, p in self.model.named_parameters() if 'gate_alphas' in n)


MaPLe.train = measured_train
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
sys.argv = ['train.py', *sys.argv[2:]]
metrics['argv'] = sys.argv.copy()
metrics['cuda_device'] = torch.cuda.get_device_name()
metrics['status'] = 'failed'
try:
    runpy.run_path('train.py', run_name='__main__')
    metrics['status'] = 'success'
finally:
    torch.cuda.synchronize()
    metrics['process_seconds'] = time.perf_counter() - start
    metrics['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    metrics['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('x') as f:
        json.dump(metrics, f, indent=2)
