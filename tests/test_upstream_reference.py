import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('upstream_reference', ROOT/'scripts/maple/run_upstream_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class UpstreamReferenceTests(unittest.TestCase):
    def test_reference_rejects_other_checkpoint_families(self):
        reference.validate_reference_family({'prompt_learner.ctx': None})
        for key in ('prompt_learner.gate_alphas.0', 'prompt_learner.gate_alphas',
                    'prompt_learner._gate_schema', 'prompt_learner._frozen_projection_schema'):
            with self.assertRaisesRegex(ValueError, 'cannot load'):
                reference.validate_reference_family({key: None})

    def test_only_registration_removed_and_corrupt_fixture_rejected(self):
        fixture = ROOT/'tests/fixtures/maple_upstream.py'
        original = ast.parse(fixture.read_text(encoding='utf-8'))
        actual = reference.source_tree(fixture)
        trainer = next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='MaPLe')
        trainer.decorator_list = []
        self.assertEqual(ast.dump(original),ast.dump(actual))
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder)/'changed.py'
            changed.write_bytes(fixture.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'checksum'):
                reference.source_tree(changed)


if __name__ == '__main__':
    unittest.main()

