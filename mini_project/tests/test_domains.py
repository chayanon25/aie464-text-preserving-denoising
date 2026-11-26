"""Train/val loading per source domain for the cross-domain benchmark."""
from pathlib import Path

import pytest

from src.degrade import make_augraphy_split
from src.domains import AUGRAPHY_TRAIN_DIR, DOMAINS, load_domain_train
from src.smoke import smoke_data

MANIFEST = Path(__file__).resolve().parents[1] / 'outputs' / 'phase1' / 'split_manifest.json'


def test_domains_constant():
    assert DOMAINS == ('kaggle', 'augraphy', 'dibco')


def test_unknown_domain_raises(tmp_path):
    with pytest.raises(ValueError):
        load_domain_train('mnist', tmp_path, MANIFEST)


def test_missing_augraphy_dir_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match='make_augraphy_split'):
        load_domain_train('augraphy', tmp_path, MANIFEST)


def test_augraphy_domain_loads_generated_split(kaggle_root, tmp_path):
    for split in ('train', 'val'):   # val-sized stand-in keeps the test fast
        make_augraphy_split(kaggle_root, MANIFEST, tmp_path / AUGRAPHY_TRAIN_DIR / split, 'val')
    data = load_domain_train('augraphy', tmp_path, MANIFEST)
    assert len(data['train']) == 20 and len(data['val']) == 20


def test_smoke_data_has_augraphy():
    assert set(smoke_data()['augraphy']) == {'train', 'val'}
