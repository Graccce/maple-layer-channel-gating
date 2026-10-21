"""Validate a local official DTD archive and fixed CoOp split before adoption."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

ARCHIVE_MD5 = 'fff73e5086ae6bdbea199a49dfb8a4c1'
SPLIT_SHA256 = 'f26ebeccd2b58e68d70f07a0dc39fe49bf7b69024bac34396b954dfd87969c38'
SPLIT_NAME = 'split_zhou_DescribableTextures.json'


def digest(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def validate_members(members):
    for member in members:
        path = PurePosixPath(member.name)
        if (not path.parts or path.is_absolute() or '..' in path.parts or
                path.parts[0] != 'dtd' or not (member.isfile() or member.isdir()) or
                '\\' in member.name):
            raise ValueError(f'Unsafe archive member: {member.name}')


def validate_split(split, images):
    if set(split) != {'train', 'val', 'test'}:
        raise ValueError('Expected train/val/test split keys')
    classes = sorted({path.split('/')[0] for path in images})
    seen = set()
    counts = {}
    for subset, rows in split.items():
        counts[subset] = len(rows)
        for row in rows:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError('Expected [relative_image_path, label, classname]')
            name, label, classname = row
            if (not isinstance(name, str) or name not in images or name in seen or
                    type(label) is not int or not 0 <= label < len(classes) or
                    classname != classes[label] or name.split('/')[0] != classname):
                raise ValueError('Missing/duplicate image or inconsistent class label')
            seen.add(name)
    if seen != images:
        raise ValueError('Split must cover every image exactly once')
    return counts


def install_split(folder, source):
    # MaPLe looks up this exact filename even when the downloaded file was renamed.
    destination = folder / SPLIT_NAME
    with destination.open('xb') as handle:
        handle.write(source.read_bytes())
    return destination


def prepare(archive, split_path, data_root):
    target = data_root / 'dtd'
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing dataset: {target}')
    if digest(archive, 'md5') != ARCHIVE_MD5:
        raise ValueError('DTD archive does not match the official torchvision MD5')
    if digest(split_path, 'sha256') != SPLIT_SHA256:
        raise ValueError('Split differs from the pinned CoOp public split')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    data_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='dtd_prepare_', dir=data_root))
    # Preserve staging on failure for inspection; never delete existing data.
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        validate_members(members)
        tar.extractall(staging, members=members)
    image_root = staging / 'dtd/images'
    files = list(image_root.glob('*/*.jpg'))
    images = {path.relative_to(image_root).as_posix() for path in files}
    distribution = Counter(path.split('/')[0] for path in images)
    if len(images) != 5640 or len(distribution) != 47 or set(distribution.values()) != {120}:
        raise ValueError('Expected 47 classes, 120 images per class, 5640 total')
    counts = validate_split(split, images)
    if counts != {'train': 2820, 'val': 1128, 'test': 1692}:
        raise ValueError('Unexpected fixed split counts')
    from PIL import Image
    for path in files:
        with Image.open(path) as image:
            image.verify()
    install_split(staging / 'dtd', split_path)
    facts = dict(status='success', archive_md5=ARCHIVE_MD5,
                 archive_sha256=digest(archive, 'sha256'), split_sha256=SPLIT_SHA256,
                 images=5640, classes=47, split_counts=counts,
                 all_images_verified=True,
                 official_archive_url='https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz',
                 split_source='https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#dtd')
    (staging / 'dtd/provenance.json').write_text(json.dumps(facts, indent=2)+'\n', encoding='utf-8')
    if target.exists():
        raise FileExistsError('Dataset target appeared during validation; staging preserved')
    os.rename(staging / 'dtd', target)
    return facts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--split', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.split, args.data_root), indent=2))
