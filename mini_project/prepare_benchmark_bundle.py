"""Copy-only ZIP for one Drive upload. No network, secrets or training.

Usage: python3 mini_project/prepare_benchmark_bundle.py --include-test
Default includes only train/val; --include-test copies the reserved test files
without decoding or evaluating them. Existing archives are never overwritten.
"""
import argparse
from pathlib import Path
import sys
import zipfile


def prepare_bundle(project_dir, archive, include_test=False):
    project_dir, archive = Path(project_dir).resolve(), Path(archive).resolve()
    sys.path.insert(0, str(project_dir / 'src'))
    from benchmark import load_manifest
    manifest = load_manifest(project_dir)
    # Explicit allowlist: never traverse unrelated files or credentials.
    files = ['src/dataset.py', 'src/evaluate.py', 'src/benchmark.py', 'src/train.py', 'src/losses.py',
             'src/models/shallow_cnn.py', 'src/models/dncnn.py', 'src/models/unet.py',
             'outputs/phase1/split_manifest.json', 'notebooks/03_final_benchmark_colab.ipynb']
    splits = ('train', 'val', 'test') if include_test else ('train', 'val')
    for split in splits:
        for page in manifest['splits'][split]:
            files.extend(f'data/raw/{folder}/{page}' for folder in ('train', 'train_cleaned'))
    for relative in files:
        if not (project_dir / relative).is_file():
            raise FileNotFoundError(relative)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, mode='x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in files:
            bundle.write(project_dir / relative, f'mini_project/{relative}')
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None or len(bundle.namelist()) != len(files):
            raise RuntimeError('ZIP integrity check failed')
    print(f'Created {archive}: {len(files)} files; include_test={include_test}. No upload performed.')
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    prepare_bundle(root, args.output or root / 'outputs/benchmark_bundle.zip', args.include_test)
