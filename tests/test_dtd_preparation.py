import importlib.util
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_dtd', Path(__file__).resolve().parents[1] /
                                            'scripts/maple/prepare_dtd.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DTDPreparationTests(unittest.TestCase):
    def test_archive_rejects_traversal_links_and_wrong_root(self):
        module.validate_members([tarfile.TarInfo('dtd/images/banded/image.jpg')])
        for name, kind in [('dtd/../escape', tarfile.REGTYPE), ('/dtd/file', tarfile.REGTYPE),
                           ('other/file', tarfile.REGTYPE), ('dtd/link', tarfile.SYMTYPE)]:
            member = tarfile.TarInfo(name)
            member.type = kind
            with self.assertRaises(ValueError):
                module.validate_members([member])

    def test_split_coverage_overlap_and_labels(self):
        images = {'a/one.jpg', 'a/two.jpg', 'b/three.jpg'}
        data = {'train': [['a/one.jpg', 0, 'a']], 'val': [['a/two.jpg', 0, 'a']],
                'test': [['b/three.jpg', 1, 'b']]}
        self.assertEqual(module.validate_split(data, images), dict(train=1, val=1, test=1))
        data['test'] = [['a/one.jpg', 0, 'a']]
        with self.assertRaises(ValueError):
            module.validate_split(data, images)
        data['test'] = [['b/three.jpg', 0, 'b']]
        with self.assertRaises(ValueError):
            module.validate_split(data, images)

    def test_corrupt_archive_and_existing_dataset_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'invalid.tar.gz'
            archive.write_bytes(b'invalid')
            with self.assertRaises(ValueError):
                module.prepare(archive, root / 'absent.json', root)
            (root / 'dtd').mkdir()
            with self.assertRaises(FileExistsError):
                module.prepare(archive, root / 'absent.json', root)

    def test_renamed_download_installs_canonical_split_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'dataset'
            folder.mkdir()
            source = root / 'renamed_download.json'
            source.write_bytes(b'{"train": [], "val": [], "test": []}')
            destination = module.install_split(folder, source)
            self.assertEqual(destination.name, 'split_zhou_DescribableTextures.json')
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            with self.assertRaises(FileExistsError):
                module.install_split(folder, source)


if __name__ == '__main__':
    unittest.main()
