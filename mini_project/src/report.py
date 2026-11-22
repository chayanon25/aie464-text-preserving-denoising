"""Markdown tables for the improvement report, built from the per-image eval JSONs."""
import json
from pathlib import Path

import numpy as np

from .stats import paired_wilcoxon

# Rows shown per table; names missing from the eval dir are skipped.
IN_DOMAIN = ['midterm_cnn_l1', 'kd_cnn', 'midterm_dncnn_lite', 'dncnn_nobn_aug', 'midterm_unet_l1',
             'midterm_unet_edge', 'unet_edge_aug', 'unet_edge_aug_s43', 'unet_edge_aug_s44', 'unet_aug_l0',
             'unet_aug_l05', 'nafnet_zeroshot', 'nafnet_ft', 'classical_noisy', 'classical_median3',
             'classical_sauvola']
FINETUNE = ['unet_edge_aug', 'ft_full', 'ft_freeze_enc', 'ft_lora4', 'ft_lora8', 'nafnet_ft',
            'nafnet_ft_dibco_full', 'nafnet_ft_dibco_lora8']
DIBCO = ['classical_noisy', 'classical_sauvola', 'midterm_unet_edge', 'unet_edge_aug', 'nafnet_zeroshot',
         'nafnet_ft', 'ft_full', 'ft_freeze_enc', 'ft_lora4', 'ft_lora8', 'nafnet_ft_dibco_full',
         'nafnet_ft_dibco_lora8']
# On DIBCO every output is Otsu-binarized before scoring, so 'noisy' already is Otsu on the raw page.
LABELS = {'classical_noisy': 'noisy input'}
DIBCO_LABELS = {'classical_noisy': 'Otsu on raw input'}
# Pilots are throughput probes, and classical_otsu equals classical_noisy on DIBCO (both Otsu-scored).
HIDDEN = {'pilot_unet_edge_aug', 'nafnet_short_ft'}
PAIRS = [
    ('unet_edge_aug', 'midterm_unet_edge', 'kaggle_test', 'psnr'),
    ('unet_aug_l05', 'midterm_unet_edge', 'kaggle_test', 'psnr'),
    ('dncnn_nobn_aug', 'midterm_dncnn_lite', 'kaggle_test', 'psnr'),
    ('kd_cnn', 'midterm_cnn_l1', 'kaggle_test', 'psnr'),
    ('nafnet_ft', 'unet_edge_aug', 'kaggle_test', 'psnr'),
    ('unet_edge_aug', 'midterm_unet_edge', 'augraphy_shift', 'psnr'),
    ('ft_full', 'unet_edge_aug', 'dibco2017', 'fm'),
    ('ft_lora8', 'ft_full', 'dibco2017', 'fm'),
    ('nafnet_ft_dibco_full', 'ft_full', 'dibco2017', 'fm'),
]


def load_evals(eval_dir):
    evals = {}
    for path in sorted(Path(eval_dir).glob('*.json')):
        data = json.loads(path.read_text())
        if 'per_image' in data:
            evals[data.get('name', path.stem)] = data['per_image']
    return evals


def scores(per_image, testset, metric, group=None):
    return {r['id']: r[metric] for r in per_image
            if r['testset'] == testset and (group is None or r.get('group') == group)}


def _mean(per_image, testset, metric, group=None):
    vals = [v for v in scores(per_image, testset, metric, group).values() if v == v]  # drop NaN
    return float(np.mean(vals)) if vals else float('nan')


def _table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def _f(x, nd=2):
    return '-' if x != x else f'{x:.{nd}f}'


def _gray_table(evals, names, testset):
    rows = [[LABELS.get(n, n), _f(_mean(evals[n], testset, 'psnr')), _f(_mean(evals[n], testset, 'ssim'), 3),
             _f(_mean(evals[n], testset, 'text_psnr'))] for n in names if n in evals]
    return _table(['model', 'PSNR (dB)', 'SSIM', 'text PSNR (dB)'], rows)


def build_tables(eval_dir, latency_path=None, pairs=None) -> dict:
    evals = load_evals(eval_dir)
    extra = [n for n in evals if n not in IN_DOMAIN + FINETUNE + DIBCO + list(HIDDEN)]
    t = {'in_domain': _gray_table(evals, IN_DOMAIN + extra, 'kaggle_test'),
         'augraphy_shift': _gray_table(evals, IN_DOMAIN + extra, 'augraphy_shift')}

    rows = []
    for n in [n for n in DIBCO + extra if n in evals and n != 'classical_otsu']:
        row = [DIBCO_LABELS.get(n, n)]
        for group in (None, 'printed', 'handwritten'):
            row.append(_f(_mean(evals[n], 'dibco2017', 'fm', group)))
        row += [_f(_mean(evals[n], 'dibco2017', 'pfm')), _f(_mean(evals[n], 'dibco2017', 'psnr')),
                _f(_mean(evals[n], 'dibco2017', 'drd'))]
        rows.append(row)
    t['dibco2017'] = _table(['model', 'FM all', 'FM printed', 'FM handwritten', 'pFM', 'PSNR', 'DRD (lower better)'],
                            rows)

    rows = [[n, _f(_mean(evals[n], 'dibco2017', 'fm')), _f(_mean(evals[n], 'dibco2017', 'drd')),
             _f(_mean(evals[n], 'kaggle_test', 'psnr'))] for n in FINETUNE if n in evals]
    t['finetune'] = _table(['model', 'DIBCO FM', 'DIBCO DRD', 'Kaggle PSNR (forgetting)'], rows)

    latency = json.loads(Path(latency_path).read_text()) if latency_path and Path(latency_path).exists() else {}
    rows = [[n, f"{v['params']:,}", _f(v['median_ms'], 1), _f(v['peak_vram_mb'], 0)] for n, v in latency.items()
            if n not in HIDDEN]
    t['efficiency'] = _table(['model', 'params', 'median ms (540x420, T4)', 'peak VRAM MB'], rows)

    rows = []
    for a, b, testset, metric in (PAIRS if pairs is None else pairs):
        if a not in evals or b not in evals:
            continue
        sa, sb = scores(evals[a], testset, metric), scores(evals[b], testset, metric)
        ids = sorted(set(sa) & set(sb))
        r = paired_wilcoxon([sa[i] for i in ids], [sb[i] for i in ids])
        p = '<0.001' if r['pvalue'] < 0.001 else f"{r['pvalue']:.3f}"
        rows.append([a, b, testset, metric, r['n'], _f(r['median_diff']), p])
    t['wilcoxon'] = _table(['A', 'B', 'test set', 'metric', 'n', 'median(A-B)', 'p (two-sided)'], rows)
    return t
