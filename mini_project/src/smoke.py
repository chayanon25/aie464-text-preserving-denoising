"""Tiny synthetic pages for the notebook smoke run (no real data needed)."""
import numpy as np


def _page(rng, height, width, binary_gt):
    clean = np.ones((height, width), np.float32)
    for top in range(8, height - 12, 14):
        for left in range(6, width - 10, 9):
            if rng.random() < 0.7:
                clean[top:top + 7, left:left + 2 + int(rng.integers(5))] = 0.0
    noise = rng.normal(0, 0.08, clean.shape).astype(np.float32)
    stain = 0.25 * np.exp(-((np.arange(width)[None] - width / 2) ** 2) / (width ** 2 / 8)).astype(np.float32)
    noisy = np.clip(clean * 0.9 + 0.05 + noise - stain, 0, 1).astype(np.float32)
    return noisy, clean if binary_gt else np.clip(clean + 0.02 * (clean > 0), 0, 1)


def make_smoke_samples(n, group='synthetic', binary_gt=False, seed=0, height=96, width=128):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        noisy, clean = _page(rng, height, width, binary_gt)
        samples.append({'id': f'{group}_{i}', 'noisy': noisy, 'clean': clean.astype(np.float32),
                        'group': group if not binary_gt else ('printed' if i % 2 else 'handwritten')})
    return samples


def smoke_data():
    """Train/val sets for every source domain plus small test sets."""
    kaggle = {'train': make_smoke_samples(3, seed=1), 'val': make_smoke_samples(2, seed=2)}
    dibco = {'train': make_smoke_samples(3, binary_gt=True, seed=3), 'val': make_smoke_samples(2, binary_gt=True, seed=4)}
    augraphy = {'train': make_smoke_samples(3, seed=8), 'val': make_smoke_samples(2, seed=9)}
    testsets = {'kaggle_test': make_smoke_samples(2, seed=5), 'augraphy_shift': make_smoke_samples(2, seed=6),
                'dibco2017': make_smoke_samples(2, binary_gt=True, seed=7)}
    return {'kaggle': kaggle, 'augraphy': augraphy, 'dibco': dibco, 'testsets': testsets}
