"""Train/val data for each source domain of the cross-domain benchmark.

Test sets are never loaded here: they come from eval_suite.load_testsets so
every model is scored on the same fixed images.
"""
from pathlib import Path

from .data_external import dibco_finetune_split
from .dataset import load_kaggle_split
from .degrade import load_shift_testset

DOMAINS = ('kaggle', 'augraphy', 'dibco')
AUGRAPHY_TRAIN_DIR = 'external/augraphy_train'   # relative to data_root; holds train/ and val/


def load_domain_train(domain, data_root, manifest):
    """Return {'train': samples, 'val': samples} for one source domain."""
    data_root = Path(data_root)
    if domain == 'kaggle':
        root = data_root / 'raw'
        return {split: load_kaggle_split(root, manifest, split) for split in ('train', 'val')}
    if domain == 'dibco':
        return dibco_finetune_split(data_root / 'external' / 'dibco', seed=42)
    if domain == 'augraphy':
        base = data_root / AUGRAPHY_TRAIN_DIR
        missing = [s for s in ('train', 'val') if not (base / s / 'manifest.json').exists()]
        if missing:
            raise FileNotFoundError(
                f'Augraphy {missing} split missing under {base}; generate it with make_augraphy_split')
        return {split: load_shift_testset(base / split) for split in ('train', 'val')}
    raise ValueError(f'Unknown domain {domain!r}; expected one of {DOMAINS}')
