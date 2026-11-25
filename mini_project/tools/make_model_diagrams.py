"""Architecture diagrams for the comparison models (DnCNN-lite, ShallowCNN, NAFNet).

Same visual style as the SmallUNet diagram in make_figures.py. Shapes are for a
256x256 grayscale input and follow src/models/{dncnn,shallow_cnn,nafnet}.py.

    python tools/make_model_diagrams.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2] / 'docs' / 'figures'
BLUE, ORANGE, GREEN, GRAY = '#4c78a8', '#f58518', '#54a24b', '#9d9d9d'


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def _bars(ax, blocks):
    """blocks: list of (label, shape_text, spatial_size, color)."""
    for x, (label, shape, size, color) in enumerate(blocks):
        height = 0.6 + 2.4 * size / 256
        ax.add_patch(plt.Rectangle((x - 0.3, -height / 2), 0.6, height, color=color, alpha=0.8))
        ax.text(x, height / 2 + 0.15, label, ha='center', fontsize=9, weight='bold')
        ax.text(x, -height / 2 - 0.35, shape, ha='center', fontsize=8)
    for x in range(len(blocks) - 1):
        ax.annotate('', xy=(x + 0.68, 0), xytext=(x + 0.32, 0),
                    arrowprops=dict(arrowstyle='->', color='gray'))


def _residual(ax, n, y=2.1, text='residual: output = input - predicted noise'):
    ax.plot([0, 0, n - 1], [1.75, y, y], color='gray', lw=1)
    ax.annotate('', xy=(n - 1, 1.75), xytext=(n - 1, y), arrowprops=dict(arrowstyle='->', color='gray'))
    ax.text((n - 1) / 2, y + 0.12, text, ha='center', fontsize=8, color='gray')


def _finish(ax, n, title, note=None):
    if note:
        ax.text(n - 0.5, -2.2, note, ha='right', fontsize=8)
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-2.4, 2.6); ax.axis('off')
    ax.set_title(title, fontsize=10)


def fig_dncnn():
    blocks = [('input', '1x256x256', 256, GRAY),
              ('conv+ReLU', '32x256x256', 256, BLUE),
              ('5 x [conv+BN+ReLU]', '32x256x256', 256, BLUE),
              ('conv', 'noise 1x256x256', 256, ORANGE),
              ('output', '1x256x256', 256, GREEN)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    _bars(ax, blocks)
    _residual(ax, len(blocks))
    _finish(ax, len(blocks), 'DnCNN-lite (width 32, 7 conv layers, 47,009 params), 256x256 input',
            'all 3x3 conv, no downsampling; no-BN variant has 46,849 params')
    _save(fig, 'dncnn_diagram.png')


def fig_shallow():
    blocks = [('input', '1x256x256', 256, GRAY),
              ('conv+ReLU', '16x256x256', 256, BLUE),
              ('3 x [conv+ReLU]', '16x256x256', 256, BLUE),
              ('conv+Sigmoid', '1x256x256', 256, ORANGE),
              ('output', '1x256x256', 256, GREEN)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _bars(ax, blocks)
    _finish(ax, len(blocks), 'ShallowCNN (width 16, 5 conv layers, 7,265 params), 256x256 input',
            'predicts the clean image directly (no residual); also the KD student')
    ax.set_ylim(-2.4, 2.0)
    _save(fig, 'shallowcnn_diagram.png')


def fig_nafnet():
    blocks = [('gray->RGB', '3x256x256', 256, GRAY),
              ('enc1 (2)', '32x256x256', 256, BLUE),
              ('enc2 (2)', '64x128x128', 128, BLUE),
              ('enc3 (4)', '128x64x64', 64, BLUE),
              ('enc4 (8)', '256x32x32', 32, BLUE),
              ('middle (12)', '512x16x16', 16, BLUE),
              ('dec4 (2)', '256x32x32', 32, ORANGE),
              ('dec3 (2)', '128x64x64', 64, ORANGE),
              ('dec2 (2)', '64x128x128', 128, ORANGE),
              ('dec1 (2)', '32x256x256', 256, ORANGE),
              ('RGB->gray', '1x256x256', 256, GREEN)]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    _bars(ax, blocks)
    for k, (a, b) in enumerate(((4, 6), (3, 7), (2, 8), (1, 9))):
        y = 1.9 + 0.25 * k
        ax.plot([a, a, b], [y - 0.2, y, y], color='gray', lw=1)
        ax.annotate('', xy=(b, y - 0.2), xytext=(b, y), arrowprops=dict(arrowstyle='->', color='gray'))
    ax.text(5, 2.85, 'skip connections (add)', ha='center', fontsize=8, color='gray')
    _finish(ax, len(blocks),
            'NAFNet (width 32, pretrained on SIDD, 29,159,715 params), 256x256 input',
            '(n) = number of NAFBlocks; output = input + predicted residual; gray input repeated to 3 channels, output averaged back')
    ax.set_ylim(-2.4, 3.1)
    _save(fig, 'nafnet_diagram.png')


if __name__ == '__main__':
    fig_dncnn()
    fig_shallow()
    fig_nafnet()
