"""Augraphy controlled-shift test set: test split only, deterministic."""
import json
from pathlib import Path

from src.degrade import load_shift_testset, make_shift_testset

MANIFEST = Path(__file__).resolve().parents[1] / 'outputs' / 'phase1' / 'split_manifest.json'


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
