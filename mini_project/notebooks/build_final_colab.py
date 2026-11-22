"""Generate notebooks/final_colab.ipynb from code (readable diffs, testable smoke run).

Usage:  python notebooks/build_final_colab.py        # rewrites final_colab.ipynb
The notebook only orchestrates; all logic lives in mini_project/src/.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_PATH = Path(__file__).resolve().parent / 'final_colab.ipynb'


def _config(smoke, local_repo):
    return f'''# 1. Config: every setting lives here
DRIVE_ROOT = '/content/drive/MyDrive/AIE464_runs'
REPO_URL = 'https://github.com/chayanon25/aie464-text-preserving-denoising.git'
GIT_SHA = 'main'          # set to the exact commit sha before running jobs
JOB_ID = None             # None = claim the next pending job from runs/jobs.json
OWNER = 'account-1'       # a label for this Colab account
SMOKE = {smoke!r}            # True = tiny synthetic CPU run, no Drive
MAX_MINUTES = 150         # pause safely before Colab kills the session
LOCAL_REPO = {str(local_repo) if local_repo else None!r}

# Training budget per job: (epochs, samples_per_epoch). Filled in after the pilot
# (plan Task 12); only jobs listed here are registered. Pilot budgets use equal
# optimizer steps: 2000/8 (U-Net) == 1000/4 (NAFNet) == 250 steps.
BUDGETS = {{
    'pilot_unet_edge_aug': (1, 2000),
    'nafnet_short_ft': (1, 1000),
}}

# "Run on any folder" (cell 9): leave INPUT_DIR = None to skip.
INPUT_DIR = None
GT_DIR = None
OUTPUT_DIR = None
MODEL_REF = 'unet_edge_aug'   # job_id, 'midterm:<exp>' or 'nafnet:sidd'
'''


SETUP = '''# 2. Mount Drive, clone the repo at GIT_SHA, put mini_project/ on sys.path
import os, subprocess, sys
from pathlib import Path
if SMOKE:
    MINI = Path(LOCAL_REPO)
    RUNS_ROOT = Path('runs').resolve()
else:
    from google.colab import drive
    drive.mount('/content/drive')
    if not Path('/content/repo').exists():
        subprocess.run(['git', 'clone', '-q', REPO_URL, '/content/repo'], check=True)
    subprocess.run(['git', '-C', '/content/repo', 'fetch', '-q', 'origin'], check=True)
    subprocess.run(['git', '-C', '/content/repo', 'checkout', '-q', GIT_SHA], check=True)
    MINI = Path('/content/repo/mini_project')
    RUNS_ROOT = Path(DRIVE_ROOT) / 'runs'
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MINI))
SHA = subprocess.run(['git', '-C', str(MINI), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
print('code sha:', SHA, '| runs:', RUNS_ROOT)
'''

INSTALL = '''# 3. Extra packages (Colab already ships torch; never reinstall it)
if not SMOKE:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'augraphy', 'scikit-image'], check=True)
'''

DATA = '''# 4. Data: unzip the Drive bundle, run the fast test suite, define data loaders
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if SMOKE:
    from src.smoke import smoke_data
    SMOKE_SETS = smoke_data()
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
    _t = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-m', 'not slow'], cwd=MINI, capture_output=True, text=True)
    print(_t.stdout[-4000:], _t.stderr[-2000:])
    assert _t.returncode == 0, 'fast test suite failed (see output above)'

def get_job_data(cfg):
    if SMOKE:
        return SMOKE_SETS[cfg.data]
    if cfg.data == 'kaggle':
        from src.dataset import load_kaggle_split
        manifest = MINI / 'outputs' / 'phase1' / 'split_manifest.json'
        return {s: load_kaggle_split(DATA_ROOT / 'raw', manifest, s) for s in ('train', 'val')}
    from src.data_external import dibco_finetune_split
    return dibco_finetune_split(DATA_ROOT / 'external' / 'dibco', seed=42)

def get_testsets():
    if SMOKE:
        return SMOKE_SETS['testsets']
    from src.eval_suite import load_testsets
    return load_testsets(DATA_ROOT / 'raw', MINI / 'outputs' / 'phase1' / 'split_manifest.json',
                         DATA_ROOT / 'external' / 'augraphy_shift', DATA_ROOT / 'external' / 'dibco',
                         required=('kaggle_test', 'augraphy_shift', 'dibco2017'))
print('device:', DEVICE)
'''

REGISTER = '''# 5. Register budgeted jobs in the shared registry (idempotent)
from src.experiments import JOBS, with_budget
from src.jobs import JobRegistry
REGISTRY = JobRegistry(RUNS_ROOT / 'jobs.json')
SMOKE_OVERRIDES = dict(batch_size=2, patch=64, amp=False)

def budgeted(job_id):
    if SMOKE:
        return with_budget(job_id, 1, 4, git_sha=SHA, **SMOKE_OVERRIDES)
    epochs, samples = BUDGETS[job_id]
    return with_budget(job_id, epochs, samples, git_sha=SHA)

for job_id in (['pilot_unet_edge_aug'] if SMOKE else BUDGETS):
    REGISTRY.add(budgeted(job_id))
print(REGISTRY.summary())
'''

RUN = '''# 6. Claim and run a job; checkpoint + heartbeat every epoch, pause before MAX_MINUTES
import time
from src.train_engine import run_job
if JOB_ID:
    cfg = budgeted(JOB_ID)
    REGISTRY.add(cfg)
    REGISTRY.mark(JOB_ID, 'running', owner=OWNER)
else:
    cfg = REGISTRY.claim_next(OWNER)
if cfg is None:
    print('No pending job.')
else:
    started, result = time.time(), None
    data = get_job_data(cfg)
    while True:
        left = MAX_MINUTES - (time.time() - started) / 60
        if left <= 0:
            break
        result = run_job(cfg, RUNS_ROOT / cfg.job_id, data, DEVICE, max_minutes=left, max_epochs=1,
                         runs_root=RUNS_ROOT, on_progress=lambda: REGISTRY.heartbeat(cfg.job_id, owner=OWNER))
        REGISTRY.heartbeat(cfg.job_id, owner=OWNER)
        last = result['history'][-1] if result['history'] else {}
        print(cfg.job_id, result['status'], last)
        if result['status'] == 'done':
            REGISTRY.mark(cfg.job_id, 'done', owner=OWNER)
            break
    print('finished call with status:', result and result['status'])
'''

EVALUATE = '''# 7. Evaluate every finished run (plus midterm and classical baselines) on all test sets
import json
from src.classical import CLASSICAL
from src.eval_suite import evaluate, predict_fn, save_eval
from src.train_engine import load_trained
TESTSETS = get_testsets()
EVAL_TILE = 1024
models = {}
for job_id, status in REGISTRY.summary().items():
    if status == 'done' and (RUNS_ROOT / job_id / 'best.pt').exists():
        models[job_id] = job_id
if not SMOKE:
    for exp in ('cnn_l1', 'dncnn_lite', 'unet_l1', 'unet_edge'):
        models[f'midterm_{exp}'] = f'midterm:{exp}'
    if Path(os.environ['NAFNET_WEIGHTS']).exists():
        models['nafnet_zeroshot'] = 'nafnet:sidd'
for name, ref in models.items():
    out = RUNS_ROOT / 'eval' / f'{name}.json'
    if out.exists():
        continue
    model = load_trained(ref, RUNS_ROOT, DEVICE)
    save_eval(evaluate(predict_fn(model, DEVICE, tile=EVAL_TILE, amp=False), TESTSETS, name), out)
    print('evaluated', name)
for name, fn in CLASSICAL.items():
    out = RUNS_ROOT / 'eval' / f'classical_{name}.json'
    if not out.exists():
        save_eval(evaluate(fn, TESTSETS, f'classical_{name}'), out)
'''

LATENCY = '''# 8. Latency: warm-up 10, measure 50, median ms + peak VRAM + params (540x420 page)
from src.infer import measure_latency
latency_path = RUNS_ROOT / 'latency.json'
latency = json.loads(latency_path.read_text()) if latency_path.exists() else {}
for name, ref in models.items():
    if name not in latency:
        model = load_trained(ref, RUNS_ROOT, DEVICE)
        latency[name] = measure_latency(model, (1, 1, 540, 420), DEVICE,
                                        warmup=2 if SMOKE else 10, iters=3 if SMOKE else 50)
        latency[name]['device'] = torch.cuda.get_device_name() if DEVICE == 'cuda' else 'cpu'
latency_path.write_text(json.dumps(latency, indent=2))
print(json.dumps(latency, indent=2))
'''

ANY_FOLDER = '''# 9. Run on any folder: denoise every image, save PNGs, score against GT if given
from src.infer import run_folder
if INPUT_DIR:
    model = load_trained(MODEL_REF, RUNS_ROOT, DEVICE)
    rows = run_folder(model, INPUT_DIR, OUTPUT_DIR or f'{INPUT_DIR}_denoised', DEVICE, gt_dir=GT_DIR, tile=EVAL_TILE)
    for row in rows:
        print(row)
'''

EXPORT = '''# 10. Export a compact summary for the report (copy runs/ to outputs/final/runs/ locally)
summary = {}
for path in sorted((RUNS_ROOT / 'eval').glob('*.json')):
    summary[path.stem] = json.loads(path.read_text())['summary']
(RUNS_ROOT / 'summary.json').write_text(json.dumps({'git_sha': SHA, 'summary': summary,
                                                    'latency': latency}, indent=2))
for name, s in summary.items():
    print(name, {k: round(v.get('psnr', float('nan')), 2) for k, v in s.items() if '/' not in k})
'''


def build_notebook(smoke=False, local_repo=None):
    cells = [
        new_markdown_cell('# AIE464 Final: resumable Colab runner\n\n'
                          'Generated by `notebooks/build_final_colab.py`; edit the builder, not this file. '
                          'Run cells in order. Full training starts only after the pilot is approved.'),
        new_code_cell(_config(smoke, local_repo)),
        new_code_cell(SETUP), new_code_cell(INSTALL), new_code_cell(DATA), new_code_cell(REGISTER),
        new_code_cell(RUN), new_code_cell(EVALUATE), new_code_cell(LATENCY), new_code_cell(ANY_FOLDER),
        new_code_cell(EXPORT),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['accelerator'] = 'GPU'
    for i, cell in enumerate(nb.cells):
        cell['id'] = f'cell-{i}'
    return nb


if __name__ == '__main__':
    nbformat.write(build_notebook(), NB_PATH)
    print('wrote', NB_PATH)
