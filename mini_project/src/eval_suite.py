"""One evaluation protocol for every model on the fixed test sets.

Grayscale sets (kaggle_test, augraphy_shift): PSNR, SSIM, text-masked PSNR.
Binary sets (dibco2017): DIBCO FM / pFM / PSNR / DRD after Otsu binarization.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .data_external import load_dibco
from .dataset import load_kaggle_split
from .degrade import load_shift_testset
from .evaluate import image_metrics, text_masked_psnr
from .infer import denoise
from .metrics_binary import dibco_metrics
from .models import MIDTERM_EXPERIMENTS, build_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIDTERM_CKPT_DIR = PROJECT_ROOT / 'backups' / 'aie464_final_artifacts' / 'checkpoints'
BINARY_TESTSETS = {'dibco2017'}


def load_midterm(exp):
    """Midterm architecture with its best.pt weights (strict)."""
    path = MIDTERM_CKPT_DIR / exp / 'best.pt'
    if not path.exists():
        raise FileNotFoundError(f'Missing midterm checkpoint: {path}')
    state = torch.load(path, map_location='cpu', weights_only=False)['model_state_dict']
    model = build_model(MIDTERM_EXPERIMENTS[exp])
    model.load_state_dict(state, strict=True)
    return model


def predict_fn(model, device, tile=None, amp=False):
    model = model.to(device).eval()
    return lambda img: denoise(model, img, device, tile=tile, amp=amp)


def load_testsets(kaggle_root=None, manifest=None, shift_dir=None, dibco_root=None):
    """Load every available test set; missing data is skipped silently."""
    sets = {}
    if kaggle_root and manifest and (Path(kaggle_root) / 'train').is_dir():
        sets['kaggle_test'] = load_kaggle_split(kaggle_root, manifest, 'test')
    if shift_dir and (Path(shift_dir) / 'manifest.json').exists():
        sets['augraphy_shift'] = load_shift_testset(shift_dir)
    if dibco_root and (Path(dibco_root) / 'dibco2017').is_dir():
        sets['dibco2017'] = load_dibco(dibco_root, 2017)
    return sets


def _gray_metrics(pred, clean):
    p, c = torch.from_numpy(pred)[None, None], torch.from_numpy(clean)[None, None]
    scores = image_metrics(p, c)
    return {'psnr': float(scores['psnr'][0]), 'ssim': float(scores['ssim'][0]),
            'text_psnr': float(text_masked_psnr(p, c)[0].clamp(max=100))}


def _mean(rows, key):
    values = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.mean(values)) if values else float('nan')


def evaluate(predict, testsets, name):
    per_image = []
    for testset, samples in testsets.items():
        for sample in samples:
            pred = np.asarray(predict(sample['noisy']), dtype=np.float32)
            if pred.shape != sample['clean'].shape:
                raise ValueError(f'{name}: output shape changed on {testset}/{sample["id"]}')
            if testset in BINARY_TESTSETS:
                metrics = dibco_metrics(pred, sample['clean'] < 0.5)
            else:
                metrics = _gray_metrics(pred, sample['clean'])
            per_image.append({'testset': testset, 'id': sample['id'], 'group': sample['group'], **metrics})
    summary = {}
    for testset in testsets:
        rows = [r for r in per_image if r['testset'] == testset]
        keys = [k for k in rows[0] if k not in ('testset', 'id', 'group')] if rows else []
        summary[testset] = {k: _mean(rows, k) for k in keys} | {'n': len(rows)}
        for group in sorted({r['group'] for r in rows}):
            sub = [r for r in rows if r['group'] == group]
            summary[f'{testset}/{group}'] = {k: _mean(sub, k) for k in keys} | {'n': len(sub)}
    return {'name': name, 'per_image': per_image, 'summary': summary}


def save_eval(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
