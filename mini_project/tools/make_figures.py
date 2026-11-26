"""Figures for docs/IMPROVEMENT_REPORT.md (Task 13).

    .venv/bin/python tools/make_figures.py [--runs DIR] [--out DIR]

--runs holds the Colab runs/<job>/{best.pt,validation.json}. Figures whose inputs are missing are
skipped with a message, so the static figures can be built before the checkpoints arrive.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MINI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MINI))
from src.cross_domain import DOMAIN_LABEL, MODEL_LABEL, TESTSET_OF  # noqa: E402
from src.domains import DOMAINS  # noqa: E402
from src.eval_suite import load_testsets  # noqa: E402
from src.experiments import CROSS_DOMAIN_MODELS, cross_domain_job_id  # noqa: E402
from src.infer import denoise  # noqa: E402
from src.models.unet import SmallUNet  # noqa: E402
from src.train_engine import load_trained  # noqa: E402

DATA = MINI / 'data'
TESTSETS = dict(kaggle_root=DATA / 'raw', manifest=MINI / 'outputs' / 'phase1' / 'split_manifest.json',
                shift_dir=DATA / 'external' / 'augraphy_shift', dibco_root=DATA / 'external' / 'dibco')
# Models shown in the qualitative figures, per test set: (label, load reference).
QUAL = {
    'kaggle_test': [('midterm U-Net edge', 'midterm:unet_edge'), ('unet_aug_l05', 'unet_aug_l05'),
                    ('nafnet_ft', 'nafnet_ft')],
    'augraphy_shift': [('midterm U-Net edge', 'midterm:unet_edge'), ('unet_edge_aug', 'unet_edge_aug'),
                       ('nafnet_ft', 'nafnet_ft')],
    'dibco2017': [('unet_edge_aug (no FT)', 'unet_edge_aug'), ('ft_full', 'ft_full'),
                  ('nafnet_ft_dibco_full', 'nafnet_ft_dibco_full')],
}
CURVE_GROUPS = {
    'Kaggle (val PSNR)': ['unet_edge_aug', 'unet_aug_l0', 'unet_aug_l05', 'dncnn_nobn_aug', 'kd_cnn', 'nafnet_ft'],
    'DIBCO fine-tune (val PSNR)': ['ft_full', 'ft_freeze_enc', 'ft_lora4', 'ft_lora8', 'nafnet_ft_dibco_full',
                                   'nafnet_ft_dibco_lora8'],
}


def unet_block_shapes(size=256, width=16):
    """(C, H, W) of every SmallUNet block on a size x size input, taken with forward hooks."""
    model, shapes = SmallUNet(width).eval(), {}
    names = ['enc1', 'enc2', 'enc3', 'bridge', 'dec3', 'dec2', 'dec1', 'head']
    for name in names:
        getattr(model, name).register_forward_hook(
            lambda _m, _i, out, name=name: shapes.__setitem__(name, tuple(out.shape[1:])))
    with torch.no_grad():
        model(torch.zeros(1, 1, size, size))
    return shapes


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def _show(ax, img, title):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def fig_data_samples(sets, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for col, (name, title) in enumerate([('kaggle_test', 'Kaggle (in-domain)'),
                                         ('augraphy_shift', 'Augraphy shift'), ('dibco2017', 'DIBCO 2017')]):
        s = sets[name][0]
        _show(axes[0, col], s['noisy'], f'{title}: input')
        _show(axes[1, col], s['clean'], f'{title}: ground truth')
    _save(fig, out / 'data_samples.png')


def fig_sobel(sets, out):
    img = sets['kaggle_test'][0]['clean']
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    t = torch.from_numpy(img)[None, None]
    k = torch.from_numpy(np.stack([gx, gx.T]))[:, None]
    g = F.conv2d(F.pad(t, (1, 1, 1, 1), mode='reflect'), k)[0].numpy()
    mag = np.sqrt(g[0] ** 2 + g[1] ** 2)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))
    for ax, kern, title in ((axes[0], gx, 'Gx kernel'), (axes[1], gx.T, 'Gy kernel')):
        ax.imshow(kern, cmap='coolwarm', vmin=-2, vmax=2)
        for (r, c), v in np.ndenumerate(kern):
            ax.text(c, r, f'{v:.0f}', ha='center', va='center')
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    _show(axes[2], img, 'clean page')
    axes[3].imshow(np.abs(g[0]), cmap='magma'); axes[3].set_title('|Gx * image|', fontsize=9); axes[3].axis('off')
    axes[4].imshow(mag, cmap='magma'); axes[4].set_title('edge magnitude', fontsize=9); axes[4].axis('off')
    _save(fig, out / 'sobel_explainer.png')


def fig_unet(out):
    shapes = unet_block_shapes(256)
    order = ['enc1', 'enc2', 'enc3', 'bridge', 'dec3', 'dec2', 'dec1', 'head']
    fig, ax = plt.subplots(figsize=(12, 4.5))
    xs = np.arange(len(order))
    for x, name in zip(xs, order):
        c, h, w = shapes[name]
        height = 0.6 + 2.4 * h / 256
        ax.add_patch(plt.Rectangle((x - 0.3, -height / 2), 0.6, height,
                                   color='#4c78a8' if name.startswith(('enc', 'bridge')) else '#f58518', alpha=0.8))
        ax.text(x, height / 2 + 0.15, name, ha='center', fontsize=9, weight='bold')
        ax.text(x, -height / 2 - 0.35, f'{c}x{h}x{w}', ha='center', fontsize=8)
    for k, (a, b) in enumerate((('enc3', 'dec3'), ('enc2', 'dec2'), ('enc1', 'dec1'))):
        i, j, y = order.index(a), order.index(b), 2.0 + 0.3 * k
        ax.plot([i, i, j], [y - 0.2, y, y], color='gray', lw=1)
        ax.annotate('', xy=(j, y - 0.2), xytext=(j, y), arrowprops=dict(arrowstyle='->', color='gray'))
    ax.text(3, 2.75, 'skip connections (concat)', ha='center', fontsize=8, color='gray')
    ax.text(len(order) - 0.5, -2.2, 'output = input - predicted noise (residual)', ha='right', fontsize=8)
    ax.set_xlim(-0.6, len(order) - 0.4); ax.set_ylim(-2.4, 2.9); ax.axis('off')
    ax.set_title('SmallUNet (width 16, 487,009 params), shapes from a 256x256 forward pass', fontsize=10)
    _save(fig, out / 'unet_diagram.png')


def _load(ref, runs):
    try:
        return load_trained(ref, runs_root=runs)
    except FileNotFoundError as err:
        print('skip model', ref, '-', err)
        return None


def _crop(img, size=200):
    """Window with the most ink, so the zoom shows text rather than margin."""
    h, w = img.shape
    best, pos = -1, (0, 0)
    for y in range(0, max(1, h - size), size // 2):
        for x in range(0, max(1, w - size), size // 2):
            ink = (img[y:y + size, x:x + size] < 0.5).mean()
            if ink > best:
                best, pos = ink, (y, x)
    return slice(pos[0], pos[0] + size), slice(pos[1], pos[1] + size)


def fig_qualitative(sets, runs, out):
    for testset, entries in QUAL.items():
        models = [(label, _load(ref, runs)) for label, ref in entries]
        if any(m is None for _, m in models):
            print('skip qual', testset, '(missing checkpoints)')
            continue
        s = sets[testset][0]
        preds = [(label, denoise(m, s['noisy'], 'cpu', tile=512)) for label, m in models]
        panels = [('input', s['noisy'])] + preds + [('ground truth', s['clean'])]
        for kind, region in (('qual', (slice(None), slice(None))), ('zoom', _crop(s['clean']))):
            fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
            for ax, (label, img) in zip(axes, panels):
                _show(ax, img[region], label)
            _save(fig, out / f'{kind}_{testset}.png')


def fig_cross_domain(sets, runs, out, size=192, pad=32):
    """xd_qual.png: rows = the 6 zero-shot transfer tasks (source -> test set), columns = input,
    the 5 models trained on that source, ground truth. Each row is the inkiest window of the
    first image of that test set; models see the window plus `pad` px of context."""
    tasks = [(s, t) for s in DOMAINS for t in DOMAINS if s != t]
    refs = {(s, m): cross_domain_job_id(s, m) for s in DOMAINS for m in CROSS_DOMAIN_MODELS}
    models = {key: _load(ref, runs) for key, ref in refs.items() if key[0] in {s for s, _ in tasks}}
    missing = sorted(refs[k] for k, m in models.items() if m is None)
    if missing:
        print('skip xd_qual (missing checkpoints):', ', '.join(missing))
        return
    cols = ['input'] + [MODEL_LABEL[m] for m in CROSS_DOMAIN_MODELS] + ['ground truth']
    fig, axes = plt.subplots(len(tasks), len(cols), figsize=(1.9 * len(cols), 2.0 * len(tasks)))
    for row, (src, tgt) in enumerate(tasks):
        sample = sets[TESTSET_OF[tgt]][0]
        ys, xs = _crop(sample['clean'], size)
        h, w = sample['noisy'].shape
        y0, x0 = max(0, ys.start - pad), max(0, xs.start - pad)
        y1, x1 = min(h, ys.stop + pad), min(w, xs.stop + pad)
        inner = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
        window = sample['noisy'][y0:y1, x0:x1]
        panels = [window[inner]]
        for m in CROSS_DOMAIN_MODELS:
            panels.append(denoise(models[(src, m)], window, 'cpu')[inner])
        panels.append(sample['clean'][ys, xs])
        for col, img in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
        axes[row, 0].set_ylabel(f'{DOMAIN_LABEL[src]} -> {DOMAIN_LABEL[tgt]}', fontsize=9)
    fig.tight_layout()
    (out / 'cross_domain').mkdir(parents=True, exist_ok=True)
    _save(fig, out / 'cross_domain' / 'xd_qual.png')


def fig_curves(runs, out):
    fig, axes = plt.subplots(1, len(CURVE_GROUPS), figsize=(13, 4))
    found = 0
    for ax, (title, jobs) in zip(axes, CURVE_GROUPS.items()):
        for job in jobs:
            path = Path(runs) / job / 'validation.json'
            if not path.exists():
                continue
            hist = json.loads(path.read_text())['history']
            ax.plot([h['epoch'] for h in hist], [h['val_psnr'] for h in hist], label=job)
            found += 1
        ax.set_title(title, fontsize=10); ax.set_xlabel('epoch'); ax.set_ylabel('val PSNR (dB)')
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    if found:
        _save(fig, out / 'curves.png')
    else:
        plt.close(fig)
        print('skip curves (no validation.json under', runs, ')')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=str(MINI / 'outputs' / 'final' / 'runs'))
    ap.add_argument('--out', default=str(MINI.parent / 'docs' / 'figures'))
    ap.add_argument('--xd-only', action='store_true', help='only the cross-domain qualitative grid')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    sets = load_testsets(**TESTSETS, required=('kaggle_test', 'augraphy_shift', 'dibco2017'))
    if args.xd_only:
        fig_cross_domain(sets, args.runs, out)
        return
    fig_data_samples(sets, out)
    fig_sobel(sets, out)
    fig_unet(out)
    fig_qualitative(sets, args.runs, out)
    fig_curves(args.runs, out)
    fig_cross_domain(sets, args.runs, out)


if __name__ == '__main__':
    main()
