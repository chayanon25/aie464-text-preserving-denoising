"""Held-out Augraphy "controlled shift" test set built from Kaggle test clean pages.

Degradations (InkBleed, BleedThrough, LightingGradient, Folding) never appear
in training augmentation, so this measures robustness to unseen damage while
keeping exact clean targets.
"""
import contextlib
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

VARIANTS_PER_PAGE = 4
OPS = ['InkBleed', 'BleedThrough', 'LightingGradient', 'Folding']


def build_shift_pipeline(random_seed=None, include_folding=True):
    from augraphy import AugraphyPipeline, BleedThrough, Folding, InkBleed, LightingGradient
    ink = [InkBleed(intensity_range=(0.4, 0.7), kernel_size=(5, 5), severity=(0.3, 0.4), p=1)]
    paper = [BleedThrough(intensity_range=(0.1, 0.3), alpha=0.2, offsets=(10, 20), p=1)]
    post = [LightingGradient(max_brightness=255, min_brightness=0, mode='gaussian',
                             transparency=0.5, numba_jit=0, p=1)]
    if include_folding:
        post.append(Folding(fold_count=2, fold_noise=0.0, gradient_width=(0.1, 0.2),
                            gradient_height=(0.005, 0.01), p=1))
    return AugraphyPipeline(ink_phase=ink, paper_phase=paper, post_phase=post,
                            random_seed=random_seed)


@contextlib.contextmanager
def _isolated_cwd():
    """BleedThrough reads a random image from ./augraphy_cache/, which grows
    across calls and breaks determinism. Run each page in an empty temp cwd so
    the op always falls back to the page itself."""
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(previous)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _to_gray_uint8(image):
    image = np.asarray(image)
    if image.ndim == 3:
        image = np.asarray(Image.fromarray(image[..., :3].astype(np.uint8)).convert('L'))
    return image.astype(np.uint8)


def make_shift_testset(kaggle_root, manifest, out_dir, seed=42):
    """Write 6 unique test pages x 4 variants = 24 noisy/clean PNG pairs + manifest."""
    import augraphy
    manifest_data = json.loads(Path(manifest).read_text())
    clean_dir = Path(kaggle_root).resolve() / 'train_cleaned'
    out_dir = Path(out_dir).resolve()
    (out_dir / 'noisy').mkdir(parents=True, exist_ok=True)
    (out_dir / 'clean').mkdir(parents=True, exist_ok=True)
    groups = sorted(manifest_data['group_splits']['test'])
    files, index = [], 0
    for group in groups:
        source = sorted(manifest_data['groups'][group])[0]   # 4 pages share one clean image
        with Image.open(clean_dir / source) as image:
            clean = np.asarray(image.convert('L'), dtype=np.uint8)
        for variant in range(VARIANTS_PER_PAGE):
            random.seed(seed * 1000 + index)
            np.random.seed(seed * 1000 + index)
            with _isolated_cwd():
                degraded = _to_gray_uint8(build_shift_pipeline(seed * 1000 + index)(clean.copy()))
            if degraded.shape != clean.shape:
                raise ValueError(f'Augraphy changed page size for {source}: {degraded.shape}')
            name = f'{Path(source).stem}_v{variant}.png'
            Image.fromarray(degraded).save(out_dir / 'noisy' / name)
            Image.fromarray(clean).save(out_dir / 'clean' / name)
            files.append({'name': name, 'source': source, 'group': group, 'seed': seed * 1000 + index,
                          'sha256': _sha256(out_dir / 'noisy' / name),
                          'clean_sha256': _sha256(out_dir / 'clean' / name)})
            index += 1
    info = {'schema_version': 1, 'augraphy_version': augraphy.__version__, 'ops': OPS, 'seed': seed,
            'source_groups': groups, 'files': files}
    (out_dir / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def make_augraphy_split(kaggle_root, manifest, out_dir, split, seed=7):
    """Write an Augraphy train/val set from Kaggle train/val pages (never test).

    Folding is left out: it moves the text but not the clean target, so the
    pairs would no longer line up. Seeds (seed * 1000 + index) must differ from
    the test set's seed (42) to keep the degradations independent.
    """
    if split not in ('train', 'val'):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    import augraphy
    manifest_data = json.loads(Path(manifest).read_text())
    clean_dir = Path(kaggle_root).resolve() / 'train_cleaned'
    out_dir = Path(out_dir).resolve()
    (out_dir / 'noisy').mkdir(parents=True, exist_ok=True)
    (out_dir / 'clean').mkdir(parents=True, exist_ok=True)
    groups = sorted(manifest_data['group_splits'][split])
    files, index = [], 0
    for group in groups:
        source = sorted(manifest_data['groups'][group])[0]   # pages in a group share one clean image
        with Image.open(clean_dir / source) as image:
            clean = np.asarray(image.convert('L'), dtype=np.uint8)
        for variant in range(VARIANTS_PER_PAGE):
            page_seed = seed * 1000 + index
            random.seed(page_seed)
            np.random.seed(page_seed)
            with _isolated_cwd():
                degraded = _to_gray_uint8(
                    build_shift_pipeline(page_seed, include_folding=False)(clean.copy()))
            if degraded.shape != clean.shape:
                raise ValueError(f'Augraphy changed page size for {source}: {degraded.shape}')
            name = f'{Path(source).stem}_v{variant}.png'
            Image.fromarray(degraded).save(out_dir / 'noisy' / name)
            Image.fromarray(clean).save(out_dir / 'clean' / name)
            files.append({'name': name, 'source': source, 'group': group, 'seed': page_seed,
                          'sha256': _sha256(out_dir / 'noisy' / name),
                          'clean_sha256': _sha256(out_dir / 'clean' / name)})
            index += 1
    info = {'schema_version': 1, 'augraphy_version': augraphy.__version__,
            'ops': [op for op in OPS if op != 'Folding'], 'seed': seed, 'split': split,
            'include_folding': False, 'source_groups': groups, 'files': files}
    (out_dir / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def load_shift_testset(directory):
    """Load the shift set as samples (id, noisy, clean, group) in manifest order."""
    directory = Path(directory)
    info = json.loads((directory / 'manifest.json').read_text())
    samples = []
    for entry in info['files']:
        arrays = []
        for kind in ('noisy', 'clean'):
            with Image.open(directory / kind / entry['name']) as image:
                arrays.append(np.asarray(image.convert('L'), dtype=np.float32) / 255.0)
        samples.append({'id': entry['name'], 'noisy': arrays[0], 'clean': arrays[1], 'group': entry['group']})
    return samples
