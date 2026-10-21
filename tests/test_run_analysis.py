import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import torch

spec = importlib.util.spec_from_file_location('run_analysis', Path(__file__).resolve().parents[1] /
                                            'scripts/maple/analyze_gating_runs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RunAnalysisTests(unittest.TestCase):
    def test_counts_paired_statistics_and_scalar_channel_folding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ('none', 'scalar', 'channel'):
                for seed in (1, 2):
                    run = root / mode / f'seed{seed}'
                    run.mkdir(parents=True)
                    (run/'results.json').write_text('{}')
                    metrics = dict(status='success', trainable_parameters=100,
                                   trainer_train_seconds_including_final_eval=2,
                                   peak_allocated_bytes=2**20, peak_reserved_bytes=2**21)
                    (run/'train_metrics.json').write_text(json.dumps(metrics))
                    for subset, correct in [('base', 79+seed), ('new', 59+seed)]:
                        (run/f'test_{subset}').mkdir()
                        (run/f'test_{subset}/log.txt').write_text(f'* total: 100\n* correct: {correct}\n')
                    state = {}
                    for index in range(2):
                        prefix = f'prompt_learner.compound_prompt_projections.{index}'
                        state[prefix+'.weight'] = torch.tensor([[.2, .3], [.4, .1], [.7, .5]])
                        state[prefix+'.bias'] = torch.tensor([.1, .2, .3])
                        state[f'prompt_learner.compound_prompts_text.{index}'] = torch.tensor([[.2, .7]])
                        if mode!='none':
                            state[f'prompt_learner.gate_alphas.{index}'] = torch.tensor(.3) if mode=='scalar' else torch.tensor([.1, .2, .3])
                    cp = run/'train_base/MultiModalPromptLearner'
                    cp.mkdir(parents=True)
                    torch.save(dict(state_dict=state), cp/'model.pth.tar-5')
            report = module.summarize(root, 5)
            self.assertEqual(len(report['rows']), 6)
            self.assertAlmostEqual(report['rows'][0]['harmonic_mean'], 2*80*60/140)
            self.assertEqual(report['summary']['channel']['n'], 2)
            self.assertEqual(report['paired_comparison']['scalar']['mean_hm_delta'], 0)
            self.assertTrue(all(layer['fold_max_abs_error']<1e-12 for row in report['rows']
                                for layer in row['gate_diagnostics']))
            with self.assertRaises(ValueError):
                module.count_accuracy(root/'none/seed1/results.json')


if __name__ == '__main__':
    unittest.main()
