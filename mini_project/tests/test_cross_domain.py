"""Cross-domain 3x3 matrix, transfer gap and markdown tables."""
import json
import math

from src.cross_domain import build_matrix, matrix_markdown, transfer_gap, write_cross_domain_report

TS = {'kaggle_test', 'augraphy_shift', 'dibco2017'}


def write_eval(directory, name, **testset_psnr):
    summary = {ts: {'psnr': v, 'n': 1} for ts, v in testset_psnr.items()}
    (directory / f'{name}.json').write_text(json.dumps({'name': name, 'per_image': [], 'summary': summary}))


def test_matrix_reads_cells_and_missing_is_nan(tmp_path):
    write_eval(tmp_path, 'unet_edge_aug', kaggle_test=30.0, dibco2017=12.0)
    write_eval(tmp_path, 'xd_dibco_unet', kaggle_test=18.0, dibco2017=16.0)
    m = build_matrix(tmp_path, 'psnr')
    assert len(m) == 45
    assert m[('unet', 'kaggle', 'kaggle')] == 30.0 and m[('unet', 'dibco', 'kaggle')] == 18.0
    assert math.isnan(m[('cnn', 'augraphy', 'kaggle')])


def test_old_eval_without_fm_is_nan(tmp_path):
    write_eval(tmp_path, 'unet_edge_aug', kaggle_test=30.0)          # psnr only
    assert math.isnan(build_matrix(tmp_path, 'fm')[('unet', 'kaggle', 'kaggle')])


def test_transfer_gap_sign(tmp_path):
    write_eval(tmp_path, 'unet_edge_aug', kaggle_test=30.0, dibco2017=12.0)
    write_eval(tmp_path, 'xd_dibco_unet', kaggle_test=18.0, dibco2017=16.0)
    gap = transfer_gap(build_matrix(tmp_path, 'psnr'))
    assert gap[('unet', 'dibco', 'kaggle')] == -12.0      # 18 - 30
    assert gap[('unet', 'kaggle', 'dibco')] == -4.0       # 12 - 16
    assert ('unet', 'kaggle', 'kaggle') not in gap and len(gap) == 30


def test_markdown_marks_diagonal_and_dash(tmp_path):
    write_eval(tmp_path, 'unet_edge_aug', kaggle_test=30.0)
    md = matrix_markdown(build_matrix(tmp_path, 'psnr'), 'psnr')
    assert '**30.00**' in md and ' - ' in md and md.count('\n') >= 16


def test_non_finite_cell_does_not_crash(tmp_path):
    write_eval(tmp_path, 'xd_dibco_unet', kaggle_test=float('inf'))
    md = matrix_markdown(build_matrix(tmp_path, 'psnr'), 'psnr')
    assert 'inf' not in md


def test_write_report_has_four_tables(tmp_path):
    write_eval(tmp_path, 'unet_edge_aug', kaggle_test=30.0)
    out = tmp_path / 'out' / 'cross_domain.md'
    write_cross_domain_report(tmp_path, out)
    assert out.read_text().count('| Model | Source |') == 4
