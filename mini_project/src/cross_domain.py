"""Cross-domain benchmark tables: train on one source, test zero-shot on every test set.

Cells come from the `summary` block of each eval JSON written by eval_suite.save_eval.
A missing JSON, a missing metric (old JSON without `fm`) or a non-finite value
becomes nan and prints as '-', never an error.
"""
import json
import math
from pathlib import Path

from .domains import DOMAINS
from .experiments import CROSS_DOMAIN_MODELS, cross_domain_job_id

TESTSET_OF = {'kaggle': 'kaggle_test', 'augraphy': 'augraphy_shift', 'dibco': 'dibco2017'}
MODEL_LABEL = {'unet': 'SmallUNet (edge+aug)', 'dncnn': 'DnCNN-lite', 'cnn': 'ShallowCNN',
               'kd': 'KD-CNN', 'nafnet': 'NAFNet'}
DOMAIN_LABEL = {'kaggle': 'Kaggle', 'augraphy': 'Augraphy', 'dibco': 'DIBCO'}
METRIC_LABEL = {'psnr': 'PSNR (dB)', 'fm': 'FM (%)'}


def _summary(eval_dir, job_id):
    path = Path(eval_dir) / f'{job_id}.json'
    return json.loads(path.read_text()).get('summary', {}) if path.exists() else {}


def build_matrix(eval_dir, metric):
    """{(model, source, target): value} for all 5 x 3 x 3 cells."""
    matrix = {}
    for model in CROSS_DOMAIN_MODELS:
        for source in DOMAINS:
            summary = _summary(eval_dir, cross_domain_job_id(source, model))
            for target in DOMAINS:
                value = summary.get(TESTSET_OF[target], {}).get(metric, float('nan'))
                value = float(value) if value is not None else float('nan')
                matrix[(model, source, target)] = value if math.isfinite(value) else float('nan')
    return matrix


def transfer_gap(matrix):
    """Off-diagonal cells minus the in-domain model on that target (negative = worse)."""
    return {(m, s, t): v - matrix[(m, t, t)] for (m, s, t), v in matrix.items() if s != t}


def transfer_mean(matrix):
    """{model: mean of its 6 off-diagonal cells (source != target)}; nan if any cell is missing."""
    means = {}
    for model in CROSS_DOMAIN_MODELS:
        cells = [v for (m, s, t), v in matrix.items() if m == model and s != t]
        means[model] = sum(cells) / len(cells) if cells and all(math.isfinite(v) for v in cells) else float('nan')
    return means


def transfer_mean_markdown(eval_dir):
    """One table: mean PSNR and FM over the 6 transfer tasks per model, ranked by PSNR."""
    psnr = transfer_mean(build_matrix(eval_dir, 'psnr'))
    fm = transfer_mean(build_matrix(eval_dir, 'fm'))
    order = sorted(CROSS_DOMAIN_MODELS, key=lambda m: -psnr[m] if math.isfinite(psnr[m]) else math.inf)
    lines = [f'| Rank | Model | Mean {METRIC_LABEL["psnr"]} | Mean {METRIC_LABEL["fm"]} |', '|---|---|---|---|']
    for rank, model in enumerate(order, 1):
        lines.append(f'| {rank} | {MODEL_LABEL[model]} | {_fmt(psnr[model])} | {_fmt(fm[model])} |')
    return '\n'.join(lines) + '\n'


def _fmt(value, signed=False):
    if not math.isfinite(value):
        return '-'
    return f'{value:+.2f}' if signed else f'{value:.2f}'


def matrix_markdown(matrix, metric, gap=False):
    """Rows = model x source, columns = target test sets; in-domain cells in bold."""
    head = '| Model | Source | ' + ' | '.join(f'{DOMAIN_LABEL[t]} test' for t in DOMAINS) + ' |'
    lines = [head, '|' + '---|' * (2 + len(DOMAINS))]
    for model in CROSS_DOMAIN_MODELS:
        for source in DOMAINS:
            cells = []
            for target in DOMAINS:
                if gap and source == target:
                    cells.append('')
                    continue
                text = _fmt(matrix[(model, source, target)], signed=gap)
                cells.append(f'**{text}**' if source == target and text != '-' else text)
            lines.append(f'| {MODEL_LABEL[model]} | {DOMAIN_LABEL[source]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def write_cross_domain_report(eval_dir, out_path):
    """Write PSNR/FM matrices and their transfer gaps as one markdown file."""
    parts = ['# Cross-domain benchmark (zero-shot)\n',
             'Rows: model trained on Source. Columns: test set. Bold = in-domain. '
             '"-" = not evaluated yet.\n']
    for metric in ('psnr', 'fm'):
        matrix = build_matrix(eval_dir, metric)
        parts += [f'\n## {METRIC_LABEL[metric]}\n\n', matrix_markdown(matrix, metric)]
    for metric in ('psnr', 'fm'):
        gap = transfer_gap(build_matrix(eval_dir, metric))
        full = {k: gap.get(k, float('nan')) for k in build_matrix(eval_dir, metric)}
        parts += [f'\n## Transfer gap, {METRIC_LABEL[metric]} (cross-domain minus in-domain on the same test set)\n\n',
                  matrix_markdown(full, metric, gap=True)]
    parts += ['\n## Mean over the 6 transfer tasks (source != target, zero-shot)\n\n',
              transfer_mean_markdown(eval_dir)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(''.join(parts))
