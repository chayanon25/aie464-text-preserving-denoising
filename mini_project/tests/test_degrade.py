"""Augraphy controlled-shift test set: test split only, deterministic."""
import json
from pathlib import Path

import pytest

from src.degrade import load_shift_testset, make_augraphy_split, make_shift_testset

MANIFEST = Path(__file__).resolve().parents[1] / 'outputs' / 'phase1' / 'split_manifest.json'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_SEEDS = {42 * 1000 + i for i in range(24)}


def test_shift_testset_uses_only_test_split(kaggle_root, tmp_path):
    info = make_shift_testset(kaggle_root, MANIFEST, tmp_path, seed=42)
    test_groups = set(json.loads(MANIFEST.read_text())['group_splits']['test'])
    assert set(info['source_groups']) <= test_groups and len(info['files']) == 24
    assert len(set(info['source_groups'])) == 6


def test_shift_testset_deterministic(kaggle_root, tmp_path):
    a = make_shift_testset(kaggle_root, MANIFEST, tmp_path / 'a', seed=42)
    b = make_shift_testset(kaggle_root, MANIFEST, tmp_path / 'b', seed=42)
    assert [f['sha256'] for f in a['files']] == [f['sha256'] for f in b['files']]


def test_shift_pairs_same_size(kaggle_root, tmp_path):
    make_shift_testset(kaggle_root, MANIFEST, tmp_path, seed=42)
    samples = load_shift_testset(tmp_path)
    assert len(samples) == 24
    for s in samples:
        assert s['noisy'].shape == s['clean'].shape


def test_augraphy_split_uses_only_its_groups(kaggle_root, tmp_path):
    m = json.loads(MANIFEST.read_text())
    info = make_augraphy_split(kaggle_root, MANIFEST, tmp_path, 'val', seed=7)
    assert set(info['source_groups']) == set(m['group_splits']['val'])
    assert not set(info['source_groups']) & set(m['group_splits']['test'])
    assert len(info['files']) == 20 and info['split'] == 'val'
    assert not {f['seed'] for f in info['files']} & TEST_SEEDS
    assert 'Folding' not in info['ops'] and info['include_folding'] is False
    assert len(load_shift_testset(tmp_path)) == 20


def test_augraphy_split_refuses_test(kaggle_root, tmp_path):
    with pytest.raises(ValueError):
        make_augraphy_split(kaggle_root, MANIFEST, tmp_path, 'test')


def test_augraphy_split_deterministic(kaggle_root, tmp_path):
    a = make_augraphy_split(kaggle_root, MANIFEST, tmp_path / 'a', 'val')
    b = make_augraphy_split(kaggle_root, MANIFEST, tmp_path / 'b', 'val')
    assert [f['sha256'] for f in a['files']] == [f['sha256'] for f in b['files']]


@pytest.mark.slow
def test_augraphy_train_split_size(kaggle_root, tmp_path):
    assert len(make_augraphy_split(kaggle_root, MANIFEST, tmp_path, 'train')['files']) == 100


@pytest.mark.slow   # byte-exact only with the generating env (augraphy 8.2.6, local opencv/numpy); Colab differs
def test_existing_shift_testset_unchanged(kaggle_root, tmp_path):
    ref = PROJECT_ROOT / 'data' / 'external' / 'augraphy_shift' / 'manifest.json'
    if not ref.exists():
        pytest.skip('reference shift set not present')
    new = make_shift_testset(kaggle_root, MANIFEST, tmp_path, seed=42)
    old = json.loads(ref.read_text())
    assert [f['sha256'] for f in new['files']] == [f['sha256'] for f in old['files']]
