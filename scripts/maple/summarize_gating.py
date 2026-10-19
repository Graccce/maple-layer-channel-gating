"""Per-seed results; never substitute validation accuracy for base test accuracy."""
import json
from pathlib import Path
import re
import statistics
import sys

rows = []
for arg in sys.argv[1:]:
    run = Path(arg)
    def accuracy(subset):
        text = (run / f'test_{subset}' / 'log.txt').read_text()
        values = re.findall(r'\* accuracy:\s*([0-9.]+)%', text)
        if len(values) != 1:
            raise RuntimeError(f'Expected one completed evaluation in {run}/test_{subset}')
        return float(values[0])
    b, n = accuracy('base'), accuracy('new')
    metrics = json.loads((run / 'train_metrics.json').read_text())
    if metrics['status'] != 'success':
        raise RuntimeError(f'Training did not succeed: {run}')
    row = dict(run=str(run), base_accuracy=b, novel_accuracy=n,
               harmonic_mean=2*b*n/(b+n) if b+n else 0,
               trainable_parameters=metrics['trainable_parameters'],
               gate_parameters=metrics['gate_parameters'],
               training_seconds=metrics['trainer_train_seconds_including_final_eval'],
               peak_allocated_bytes=metrics['peak_allocated_bytes'],
               peak_reserved_bytes=metrics['peak_reserved_bytes'])
    # x mode prevents accidentally replacing an earlier result.
    result = run / 'results.json'
    if result.exists():
        if json.loads(result.read_text()) != row:
            raise RuntimeError(f'Refusing to replace different result: {result}')
    else:
        with result.open('x') as f:
            json.dump(row, f, indent=2)
    rows.append(row)
print(json.dumps(rows, indent=2))
if len(rows) > 1:
    # Only aggregate the same dataset/mode/stage from independent seeds.
    identities = {(Path(r['run']).parts[-5], Path(r['run']).parts[-3], Path(r['run']).parts[-2]) for r in rows}
    seeds = [Path(r['run']).name for r in rows]
    if len(identities) != 1 or len(set(seeds)) != len(seeds):
        raise RuntimeError('Aggregate only one dataset/mode with distinct seeds')
    print(json.dumps({key: {'mean': statistics.mean(r[key] for r in rows),
                           'sample_std': statistics.stdev(r[key] for r in rows)}
                      for key in ('base_accuracy', 'novel_accuracy', 'harmonic_mean')}, indent=2))
