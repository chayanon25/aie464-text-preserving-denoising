"""Generate notebooks/cross_domain_colab.ipynb: the 3x3 cross-domain benchmark runner.

Usage:  python notebooks/build_cross_domain_colab.py
Reuses SETUP / INSTALL / REGISTER / RUN / LATENCY from build_final_colab.py so the
training loop is the exact one that produced the Kaggle runs. Kaggle-source jobs
already done in the shared RUNS_ROOT registry are reused, not retrained.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import build_final_colab as base

NB_PATH = Path(__file__).resolve().parent / 'cross_domain_colab.ipynb'
SMOKE_JOBS = ['xd_augraphy_unet', 'xd_augraphy_kd']   # teacher then student: checks KD ordering


def _config(smoke, local_repo):
    return f'''# 1. Config: every setting lives here
DRIVE_ROOT = '/content/drive/MyDrive/AIE464_runs'   # same runs/ as the final notebook: Kaggle jobs are reused
REPO_URL = 'https://github.com/chayanon25/aie464-text-preserving-denoising.git'
GIT_SHA = 'main'          # set to the exact commit sha before running jobs
JOB_ID = None             # None = claim the next pending job from runs/jobs.json
OWNER = 'account-1'       # a label for this Colab account
SMOKE = {smoke!r}            # True = tiny synthetic CPU run, no Drive
MAX_MINUTES = 330         # pause safely before Colab kills the session (rerun to resume)
LOCAL_REPO = {str(local_repo) if local_repo else None!r}

# Budgets (epochs, samples_per_epoch) mirror the Kaggle jobs; confirm after the pilot.
# They are set in cell 2b from src.experiments.CROSS_DOMAIN_BUDGETS (teacher before KD student).
'''


BUDGET_CELL = '''# 2b. Budgets and the cross-domain job list (needs the repo on sys.path)
from src.experiments import CROSS_DOMAIN_BUDGETS, CROSS_DOMAIN_MODELS, cross_domain_job_id
from src.domains import DOMAINS
BUDGETS = dict(CROSS_DOMAIN_BUDGETS)
SMOKE_JOBS = %r
MATRIX_JOBS = [cross_domain_job_id(s, m) for s in DOMAINS for m in CROSS_DOMAIN_MODELS]
print(len(BUDGETS), 'new jobs to train |', len(MATRIX_JOBS), 'models in the matrix')
''' % SMOKE_JOBS

DATA = '''# 4. Data: unzip the Drive bundle, build the Augraphy train/val set, run the fast tests
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MANIFEST = MINI / 'outputs' / 'phase1' / 'split_manifest.json'
if SMOKE:
    from src.smoke import smoke_data
    SMOKE_SETS = smoke_data()   # no Augraphy generation in smoke mode
else:
    DATA_ROOT = Path('/content/data')
    if not (DATA_ROOT / 'raw').exists():
        subprocess.run(['unzip', '-q', '-o', f'{DRIVE_ROOT}/data/aie464_data.zip', '-d', str(DATA_ROOT)], check=True)
    os.environ['MIDTERM_CKPT_DIR'] = str(DATA_ROOT / 'checkpoints')
    os.environ['NAFNET_WEIGHTS'] = str(DATA_ROOT / 'nafnet' / 'NAFNet-SIDD-width32.pth')
    for sub in ('raw', 'external'):
        link = MINI / 'data' / sub
        link.parent.mkdir(exist_ok=True)
        if not link.exists():
            link.symlink_to(DATA_ROOT / sub)
    # Augraphy train/val from Kaggle train/val pages only (seed 7, no Folding); deterministic.
    from src.degrade import make_augraphy_split
    from src.domains import AUGRAPHY_TRAIN_DIR
    for split in ('train', 'val'):
        out = DATA_ROOT / AUGRAPHY_TRAIN_DIR / split
        if not (out / 'manifest.json').exists():
            info = make_augraphy_split(DATA_ROOT / 'raw', MANIFEST, out, split)
            print('augraphy', split, len(info['files']), 'pairs')
    _t = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-m', 'not slow'], cwd=MINI, capture_output=True, text=True)
    print(_t.stdout[-4000:], _t.stderr[-2000:])
    assert _t.returncode == 0, 'fast test suite failed (see output above)'

def get_job_data(cfg):
    if SMOKE:
        return SMOKE_SETS[cfg.data]
    from src.domains import load_domain_train
    return load_domain_train(cfg.data, DATA_ROOT, MANIFEST)

def get_testsets():
    if SMOKE:
        return SMOKE_SETS['testsets']
    from src.eval_suite import load_testsets
    return load_testsets(DATA_ROOT / 'raw', MANIFEST,
                         DATA_ROOT / 'external' / 'augraphy_shift', DATA_ROOT / 'external' / 'dibco',
                         required=('kaggle_test', 'augraphy_shift', 'dibco2017'))
print('device:', DEVICE)
'''

_OLD_SMOKE_LIST = "(['pilot_unet_edge_aug'] if SMOKE else BUDGETS)"
assert _OLD_SMOKE_LIST in base.REGISTER, 'build_final_colab.REGISTER changed; update this builder'
REGISTER = base.REGISTER.replace(_OLD_SMOKE_LIST, '(SMOKE_JOBS if SMOKE else BUDGETS)')

EVALUATE = '''# 7. Evaluate all 15 matrix models on all 3 test sets into eval_xd/ (fresh JSON with FM; eval/ untouched)
import json
from src.eval_suite import evaluate, predict_fn, save_eval
from src.train_engine import load_trained
TESTSETS = get_testsets()
EVAL_TILE = 1024
models, missing = {}, []
for job_id in MATRIX_JOBS:
    (models.__setitem__(job_id, job_id) if (RUNS_ROOT / job_id / 'best.pt').exists() else missing.append(job_id))
for name, ref in models.items():
    out = RUNS_ROOT / 'eval_xd' / f'{name}.json'
    if out.exists():
        continue
    model = load_trained(ref, RUNS_ROOT, DEVICE)
    save_eval(evaluate(predict_fn(model, DEVICE, tile=EVAL_TILE, amp=False), TESTSETS, name), out)
    print('evaluated', name)
print('not trained yet:', missing)
'''

REPORT = '''# 9. Cross-domain tables (PSNR, FM, transfer gap) -> runs/cross_domain.md
from src.cross_domain import write_cross_domain_report
write_cross_domain_report(RUNS_ROOT / 'eval_xd', RUNS_ROOT / 'cross_domain.md')
print((RUNS_ROOT / 'cross_domain.md').read_text())
'''


def build_notebook(smoke=False, local_repo=None):
    cells = [
        new_markdown_cell('# AIE464 Final: cross-domain benchmark (3 sources x 5 models)\n\n'
                          'Generated by `notebooks/build_cross_domain_colab.py`; edit the builder, not this file. '
                          'Train on Kaggle / Augraphy / DIBCO, test zero-shot on all three test sets. '
                          'Run cells in order. Full training starts only after the pilot is approved.'),
        new_code_cell(_config(smoke, local_repo)),
        new_code_cell(base.SETUP), new_code_cell(BUDGET_CELL), new_code_cell(base.INSTALL),
        new_code_cell(DATA), new_code_cell(REGISTER), new_code_cell(base.RUN),
        new_code_cell(EVALUATE), new_code_cell(base.LATENCY), new_code_cell(REPORT),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['accelerator'] = 'GPU'
    for i, cell in enumerate(nb.cells):
        cell['id'] = f'cell-{i}'
    return nb


def write_notebook(path=NB_PATH):
    path = Path(path)
    if path.exists() and any(c.get('outputs') for c in nbformat.read(path, as_version=4).cells):
        raise SystemExit(f'{path.name} has real Colab outputs; refusing to overwrite it.')
    nbformat.write(build_notebook(), path)
    print('wrote', path)


if __name__ == '__main__':
    write_notebook()
