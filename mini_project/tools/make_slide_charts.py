"""Build simple bar charts for the presentation from real Colab results.

Reads results/final_benchmark/summary.json (eval results + latency from commit 75664d9)
and writes docs/figures/slide_*.png. Numbers are never typed by hand.

Usage: .venv/bin/python tools/make_slide_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'results/final_benchmark/summary.json'
OUT = ROOT.parent / 'docs/figures'

BASE = '#9aa0a6'      # baselines / older models
MIDTERM = '#5f6b7a'   # midterm reference
OURS = '#1a73e8'      # final-project models
plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False})


def load():
    data = json.loads(SUMMARY.read_text())
    return data['summary'], data['latency']


def hbar(ax, rows, fmt='{:.2f}', xlabel=''):
    """rows: list of (label, value, color); first row drawn on top."""
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    bars = ax.barh(labels, vals, color=cols, height=0.6)
    vmax = max(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + vmax * 0.01, b.get_y() + b.get_height() / 2,
                fmt.format(v), va='center', fontsize=12)
    ax.set_xlim(0, vmax * 1.15)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=0.25)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print('wrote', OUT / name)


def chart_dataset(s, ds, metric, rows, title, xlabel, name, fmt='{:.2f}'):
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.6))
    hbar(ax, [(lab, s[k][ds][metric], c) for lab, k, c in rows], fmt, xlabel)
    ax.set_title(title, loc='left', fontsize=15, weight='bold')
    save(fig, name)


def main():
    s, lat = load()

    chart_dataset(s, 'kaggle_test', 'psnr', [
        ('noisy input', 'classical_noisy', BASE),
        ('midterm CNN', 'midterm_cnn_l1', MIDTERM),
        ('KD CNN (ours)', 'kd_cnn', OURS),
        ('midterm U-Net edge', 'midterm_unet_edge', MIDTERM),
        ('U-Net edge + aug (ours)', 'unet_edge_aug', OURS),
        ('U-Net aug, edge 0.5 (ours)', 'unet_aug_l05', OURS),
        ('NAFNet fine-tuned', 'nafnet_ft', OURS),
    ], 'Kaggle test (in-domain, 24 images): PSNR higher is better', 'PSNR (dB)',
        'slide_kaggle.png')

    chart_dataset(s, 'augraphy_shift', 'psnr', [
        ('noisy input', 'classical_noisy', BASE),
        ('midterm U-Net edge', 'midterm_unet_edge', MIDTERM),
        ('midterm DnCNN-lite', 'midterm_dncnn_lite', MIDTERM),
        ('KD CNN (ours)', 'kd_cnn', OURS),
        ('NAFNet fine-tuned', 'nafnet_ft', OURS),
        ('U-Net edge + aug (ours)', 'unet_edge_aug', OURS),
    ], 'Augraphy domain shift (24 images): PSNR higher is better', 'PSNR (dB)',
        'slide_augraphy.png')

    chart_dataset(s, 'dibco2017', 'fm', [
        ('Otsu on raw input', 'classical_otsu', BASE),
        ('Sauvola', 'classical_sauvola', BASE),
        ('U-Net edge + aug, no FT', 'unet_edge_aug', MIDTERM),
        ('NAFNet zero-shot', 'nafnet_zeroshot', MIDTERM),
        ('U-Net FT full', 'ft_full', OURS),
        ('U-Net FT LoRA r8', 'ft_lora8', OURS),
        ('NAFNet FT full', 'nafnet_ft_dibco_full', OURS),
    ], 'DIBCO 2017 (cross-dataset, 20 images): FM higher is better', 'FM (%)',
        'slide_dibco.png', fmt='{:.1f}')

    # Forgetting: same models, DIBCO FM vs Kaggle PSNR, before / after DIBCO fine-tune.
    pairs = [('U-Net\nbefore FT', 'unet_edge_aug', MIDTERM),
             ('U-Net\nFT full', 'ft_full', OURS),
             ('U-Net\nFT LoRA r8', 'ft_lora8', OURS),
             ('NAFNet\nbefore FT', 'nafnet_ft', MIDTERM),
             ('NAFNet\nFT full', 'nafnet_ft_dibco_full', OURS)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, ds, m, ttl, fmt in [(a1, 'dibco2017', 'fm', 'DIBCO 2017 FM (%): goes up', '{:.1f}'),
                                (a2, 'kaggle_test', 'psnr', 'Kaggle PSNR (dB): drops (forgetting)', '{:.2f}')]:
        vals = [s[k][ds][m] for _, k, _ in pairs]
        bars = ax.bar([p[0] for p in pairs], vals, color=[p[2] for p in pairs], width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01, fmt.format(v),
                    ha='center', fontsize=12)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.set_title(ttl, loc='left', fontsize=14, weight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='x', labelsize=11)
    save(fig, 'slide_forgetting.png')

    # KD and speed: quality vs cost.
    models = [('midterm CNN', 'midterm_cnn_l1', MIDTERM), ('KD CNN', 'kd_cnn', OURS),
              ('U-Net edge+aug', 'unet_edge_aug', OURS), ('NAFNet FT', 'nafnet_ft', OURS)]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, ttl, get, fmt in [
        (a1, 'Kaggle PSNR (dB)', lambda k: s[k]['kaggle_test']['psnr'], '{:.2f}'),
        (a2, 'Parameters', lambda k: lat[k]['params'], '{:,.0f}'),
        (a3, 'Latency on T4 (ms, 540x420)', lambda k: lat[k]['median_ms'], '{:.1f}'),
    ]:
        vals = [get(k) for _, k, _ in models]
        bars = ax.bar([m[0] for m in models], vals, color=[m[2] for m in models], width=0.6)
        if ttl == 'Parameters':
            ax.set_yscale('log')
            ax.set_ylim(1e3, max(vals) * 8)
        else:
            ax.set_ylim(0, max(vals) * 1.15)
        for b, v in zip(bars, vals):
            y = v * 1.15 if ttl == 'Parameters' else v + max(vals) * 0.01
            ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(v), ha='center', fontsize=11)
        ax.set_title(ttl, loc='left', fontsize=14, weight='bold')
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='x', labelsize=10, rotation=15)
    save(fig, 'slide_kd_speed.png')


if __name__ == '__main__':
    main()
