"""Shared job registry (runs/jobs.json on Drive) for several Colab accounts.

States: pending -> running -> done (or failed). A running job whose heartbeat
is older than STALE_MINUTES is considered abandoned and may be claimed again;
a done job is never handed out again. Writes are atomic (tmp + os.replace).
Drive offers no cross-machine lock, so accounts should start a few seconds
apart; the run directory's own last.pt keeps a double claim harmless.
"""
import json
import os
import time
from pathlib import Path

from .train_engine import JobConfig

STALE_MINUTES = 20   # run_job heartbeats within epochs (on_progress), not only between them


def now():
    return time.time()


class JobRegistry:
    def __init__(self, path):
        self.path = Path(path)

    def _read(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + '.tmp')
        tmp.write_text(json.dumps(data, indent=2) + '\n')
        os.replace(tmp, self.path)

    def add(self, cfg):
        """Register a job once; existing entries (and their status) are kept."""
        data = self._read()
        if cfg.job_id not in data:
            data[cfg.job_id] = {'config': cfg.to_dict(), 'status': 'pending', 'owner': None, 'heartbeat': None}
            self._write(data)

    def status(self, job_id):
        return self._read()[job_id]['status']

    def claim_next(self, owner):
        data = self._read()
        for job_id, entry in data.items():
            stale = (entry['status'] == 'running' and entry['heartbeat'] is not None
                     and now() - entry['heartbeat'] > STALE_MINUTES * 60)
            if entry['status'] == 'pending' or stale:
                entry.update(status='running', owner=owner, heartbeat=now())
                self._write(data)
                # Best-effort race check: re-read and confirm we still own it.
                if self._read()[job_id].get('owner') != owner:
                    continue
                return JobConfig.from_dict(entry['config'])
        return None

    def _check_owner(self, entry, job_id, owner):
        if owner is not None and entry.get('owner') not in (None, owner):
            raise PermissionError(f'{job_id} is owned by {entry["owner"]}, not {owner}')

    def heartbeat(self, job_id, owner=None):
        data = self._read()
        self._check_owner(data[job_id], job_id, owner)
        data[job_id]['heartbeat'] = now()
        self._write(data)

    def mark(self, job_id, status, owner=None):
        if status not in ('pending', 'running', 'done', 'failed'):
            raise ValueError(f'Unknown status: {status}')
        data = self._read()
        self._check_owner(data[job_id], job_id, owner)
        if owner is not None:
            data[job_id]['owner'] = owner
        data[job_id]['status'] = status
        data[job_id]['heartbeat'] = now()
        self._write(data)

    def summary(self):
        return {job_id: entry['status'] for job_id, entry in self._read().items()}
