"""Resumable training engine tests (CPU, tiny data)."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.models import build_model
from src.train_engine import JobConfig, run_job


def tiny_cfg(**kw):
    return JobConfig(job_id='t', model='cnn', data='kaggle', epochs=2, samples_per_epoch=4, batch_size=2,
                     patch=32, amp=False, augment=True, **kw)


def _sample(i, rng):
    clean = np.ones((64, 80), np.float32); clean[20:30, 10:60] = 0
    return {'id': str(i), 'noisy': np.clip(clean * 0.8 + rng.random((64, 80)).astype(np.float32) * 0.2, 0, 1),
            'clean': clean, 'group': 'x'}


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    return {'train': [_sample(i, rng) for i in range(3)], 'val': [_sample(i, rng) for i in range(3, 5)]}


@pytest.fixture
def teacher_dir(tmp_path):
    d = tmp_path / 'teacher'; d.mkdir()
    torch.save({'model': build_model('cnn').state_dict(), 'config': {'model': 'cnn'}}, d / 'best.pt')
    return d


def test_resume_equals_straight_run(tmp_path, tiny_data):
    straight = run_job(tiny_cfg(), tmp_path / 'a', tiny_data, 'cpu')
    paused = run_job(tiny_cfg(), tmp_path / 'b', tiny_data, 'cpu', max_epochs=1)   # simulated disconnect
    resumed = run_job(tiny_cfg(), tmp_path / 'b', tiny_data, 'cpu')
    sa = torch.load(tmp_path / 'a' / 'last.pt', weights_only=False)['model']
    sb = torch.load(tmp_path / 'b' / 'last.pt', weights_only=False)['model']
    assert paused['status'] == 'paused' and resumed['status'] == straight['status'] == 'done'
    assert all(torch.equal(sa[k], sb[k]) for k in sa) and len(resumed['history']) == 2


def test_atomic_save_leaves_no_tmp(tmp_path, tiny_data):
    run_job(tiny_cfg(), tmp_path, tiny_data, 'cpu')
    assert not list(tmp_path.glob('*.tmp')) and (tmp_path / 'best.pt').exists()
    assert (tmp_path / 'validation.json').exists()


def test_done_job_is_not_retrained(tmp_path, tiny_data):
    first = run_job(tiny_cfg(), tmp_path, tiny_data, 'cpu')
    again = run_job(tiny_cfg(), tmp_path, tiny_data, 'cpu')
    assert again['status'] == 'done' and len(again['history']) == len(first['history'])


def test_kd_alpha_zero_matches_plain(tmp_path, tiny_data, teacher_dir):
    a = run_job(tiny_cfg(), tmp_path / 'a', tiny_data, 'cpu')
    b = run_job(tiny_cfg(kd_teacher=str(teacher_dir), kd_alpha=0.0), tmp_path / 'b', tiny_data, 'cpu')
    assert a['history'][-1]['train_loss'] == pytest.approx(b['history'][-1]['train_loss'])


def test_missing_epochs_rejected(tmp_path, tiny_data):
    with pytest.raises(ValueError):
        run_job(replace(tiny_cfg(), epochs=None), tmp_path, tiny_data, 'cpu')


def test_init_from_job(tmp_path, tiny_data):
    run_job(tiny_cfg(), tmp_path / 'base', tiny_data, 'cpu')
    out = run_job(replace(tiny_cfg(init_from='base'), job_id='ft'), tmp_path / 'ft', tiny_data, 'cpu')
    assert out['status'] == 'done'


def test_progress_callback_called_within_epoch(tmp_path, tiny_data):
    calls = []
    run_job(tiny_cfg(), tmp_path, tiny_data, 'cpu', on_progress=lambda: calls.append(1), progress_every=1)
    assert len(calls) >= 4   # 2 steps/epoch x 2 epochs


def test_set_rng_state_accepts_tensor_not_on_cpu_bytes():
    # torch.load(map_location='cuda') moves the RNG ByteTensors to the GPU; restoring must cope.
    import torch
    from src import train_engine
    state = train_engine._rng_state()
    state['torch'] = state['torch'].to(torch.float32)  # stand-in for a non-CPU-ByteTensor copy
    train_engine._set_rng_state(state)
