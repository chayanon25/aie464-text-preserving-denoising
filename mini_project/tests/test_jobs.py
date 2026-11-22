"""Job registry tests: claiming, heartbeats, never re-running done jobs."""
import time

from src import jobs
from src.jobs import JobRegistry
from src.train_engine import JobConfig


def tiny_cfg(**kw):
    return JobConfig(job_id='t', model='cnn', data='kaggle', epochs=2, samples_per_epoch=4, batch_size=2,
                     patch=32, amp=False, **kw)


def test_registry_never_reclaims_done(tmp_path):
    r = JobRegistry(tmp_path / 'jobs.json'); r.add(tiny_cfg()); j = r.claim_next('acc1'); r.mark(j.job_id, 'done')
    assert JobRegistry(tmp_path / 'jobs.json').claim_next('acc2') is None


def test_registry_reclaims_stale_running(tmp_path, monkeypatch):
    r = JobRegistry(tmp_path / 'jobs.json'); r.add(tiny_cfg()); r.claim_next('acc1')
    monkeypatch.setattr(jobs, 'now', lambda: time.time() + 21 * 60)
    assert r.claim_next('acc2').job_id == 't'


def test_registry_fresh_running_not_reclaimed(tmp_path):
    r = JobRegistry(tmp_path / 'jobs.json'); r.add(tiny_cfg()); r.claim_next('acc1')
    assert r.claim_next('acc2') is None


def test_registry_add_idempotent_and_atomic(tmp_path):
    r = JobRegistry(tmp_path / 'jobs.json'); r.add(tiny_cfg()); r.mark('t', 'done'); r.add(tiny_cfg())
    assert r.status('t') == 'done' and not list(tmp_path.glob('*.tmp'))
