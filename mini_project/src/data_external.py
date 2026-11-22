"""External cross-dataset data: DIBCO 2017/2018/2019 (track A).

Layouts differ per year, so images are located with rglob under dibco{year}/.
GT is binary; `clean` is float {0,1} with ink = 0 (same polarity as Kaggle).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_PATH = PROJECT_ROOT / 'outputs' / 'final' / 'dibco_split.json'
FINETUNE_YEARS = (2018, 2019)
FINETUNE_VAL = 4


def _index_bmps(root: Path):
    """Map (kind, stem) -> path, where kind is 'data' or 'gt' by folder name."""
    found = {}
    for path in root.rglob('*.bmp'):
        parts = [p.lower() for p in path.relative_to(root).parts[:-1]]
        kind = 'gt' if 'gt' in parts else 'data'
        found[(kind, path.stem)] = path
    return found


def _group(year, index):
    if year == 2017:
        return 'handwritten' if index <= 10 else 'printed'
    return 'handwritten' if year == 2018 else 'mixed'


def load_dibco(root, year):
    """Load all pages of one DIBCO year as samples sorted by page number."""
    year_root = Path(root) / f'dibco{year}'
    if not year_root.is_dir():
        raise FileNotFoundError(f'Missing DIBCO folder: {year_root}')
    index = _index_bmps(year_root)
    pages = sorted(int(stem) for kind, stem in index if kind == 'data' and stem.isdigit())
    samples = []
    for i in pages:
        gt_path = index.get(('gt', f'{i}_gt')) or index.get(('gt', str(i)))
        if gt_path is None:
            raise FileNotFoundError(f'No GT for DIBCO {year} page {i}')
        with Image.open(index[('data', str(i))]) as image:
            noisy = np.array(image.convert('L'), dtype=np.float32) / 255.0
        with Image.open(gt_path) as image:
            clean = (np.array(image.convert('L')) > 127).astype(np.float32)
        if noisy.shape != clean.shape:
            raise ValueError(f'Dimension mismatch for DIBCO {year}/{i}')
        ink_ratio = float((1 - clean).mean())
        if not 0 < ink_ratio < 0.5:
            raise ValueError(f'Suspicious GT polarity for DIBCO {year}/{i}: ink ratio {ink_ratio:.3f}')
        samples.append({'id': f'{year}/{i}', 'noisy': noisy, 'clean': clean, 'group': _group(year, i)})
    return samples


def dibco_finetune_split(root, seed=42, out_path=DEFAULT_SPLIT_PATH):
    """Split DIBCO 2018 + 2019A (20 pages) into 16 train / 4 val; never touches 2017.

    The id lists are written to `out_path`; a differing existing file is never
    overwritten (same rule as save_split_manifest).
    """
    pool = [s for year in FINETUNE_YEARS for s in load_dibco(root, year)]
    order = np.random.default_rng(seed).permutation(len(pool))
    val = [pool[i] for i in order[:FINETUNE_VAL]]
    train = [pool[i] for i in order[FINETUNE_VAL:]]
    record = {'schema_version': 1, 'seed': seed, 'pool_years': list(FINETUNE_YEARS),
              'algorithm': 'NumPy default_rng(seed).permutation over pool sorted by year then page; first 4 = val',
              'train': [s['id'] for s in train], 'val': [s['id'] for s in val]}
    if out_path is not None:
        out_path = Path(out_path)
        if out_path.exists() and json.loads(out_path.read_text()) != record:
            raise ValueError(f'Existing DIBCO split differs; refusing to overwrite: {out_path}')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2) + '\n')
    return {'train': train, 'val': val}
