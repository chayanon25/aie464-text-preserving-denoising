"""Kaggle split, DIBCO loaders, fine-tune split and patch sampler tests."""
import json
from pathlib import Path

import numpy as np
import torch

from src.data_external import dibco_finetune_split, load_dibco
from src.dataset import RandomPatchDataset, load_kaggle_split

MANIFEST = Path(__file__).resolve().parents[1] / 'outputs' / 'phase1' / 'split_manifest.json'


def test_dibco_counts_and_groups(dibco_root):
    t = load_dibco(dibco_root, 2017)
    assert len(t) == 20 and sum(s['group'] == 'printed' for s in t) == 10
    assert len(load_dibco(dibco_root, 2018)) == 10 and len(load_dibco(dibco_root, 2019)) == 10


def test_dibco_gt_polarity(dibco_root):
    for year in (2017, 2018, 2019):
        for s in load_dibco(dibco_root, year):
            assert s['noisy'].shape == s['clean'].shape
            assert set(np.unique(s['clean'])) <= {0.0, 1.0}
            assert 0 < (1 - s['clean']).mean() < 0.5        # ink is minority -> polarity correct


def test_finetune_split_disjoint_from_test(dibco_root, tmp_path):
    out = tmp_path / 'dibco_split.json'
    sp = dibco_finetune_split(dibco_root, seed=42, out_path=out)
    assert len(sp['train']) == 16 and len(sp['val']) == 4
    ids = {s['id'] for s in sp['train'] + sp['val']}
    assert not any(i.startswith('2017/') for i in ids)
    assert dibco_finetune_split(dibco_root, seed=42, out_path=out)['val'][0]['id'] == sp['val'][0]['id']
    assert json.loads(out.read_text())['val'] == [s['id'] for s in sp['val']]


def test_finetune_split_refuses_overwrite(dibco_root, tmp_path):
    out = tmp_path / 'dibco_split.json'
    out.write_text(json.dumps({'train': [], 'val': []}))
    try:
        dibco_finetune_split(dibco_root, seed=42, out_path=out)
    except ValueError:
        return
    raise AssertionError('expected ValueError on differing split file')


def test_kaggle_split_matches_manifest(kaggle_root):
    assert [len(load_kaggle_split(kaggle_root, MANIFEST, s)) for s in ('train', 'val', 'test')] == [100, 20, 24]


def test_patch_dataset_pads_small_images():
    s = [{'id': 'a', 'noisy': np.zeros((191, 245), np.float32), 'clean': np.zeros((191, 245), np.float32), 'group': 'x'}]
    n, c = RandomPatchDataset(s, 256, samples_per_epoch=3)[0]
    assert n.shape == c.shape == (1, 256, 256) and n[0, -1, -1] == 1.0 and c[0, -1, -1] == 1.0


def test_patch_dataset_epoch_determinism():
    s = [{'id': 'a', 'noisy': np.random.rand(300, 400).astype(np.float32), 'clean': np.random.rand(300, 400).astype(np.float32), 'group': 'x'}]
    a, b = RandomPatchDataset(s, 64, 4, seed=42), RandomPatchDataset(s, 64, 4, seed=42)
    a.set_epoch(3); b.set_epoch(3)
    assert torch.equal(a[2][0], b[2][0])
    a.set_epoch(4)
    assert not torch.equal(a[2][0], b[2][0])
