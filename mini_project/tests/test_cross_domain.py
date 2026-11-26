"""Cross-domain 3x3 matrix, transfer gap and markdown tables."""
import json
import math

from src.cross_domain import (build_matrix, matrix_markdown, transfer_gap, transfer_mean,
                               transfer_mean_markdown, write_cross_domain_report)

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
    text = out.read_text()
    assert text.count('| Model | Source |') == 4
    assert 'Mean over the 6 transfer tasks' in text


def test_transfer_mean_uses_only_the_six_off_diagonal_cells(tmp_path):
    write_eval(tmp_path, 'unet_edge_aug', kaggle_test=30.0, augraphy_shift=16.0, dibco2017=12.0)
    write_eval(tmp_path, 'xd_augraphy_unet', kaggle_test=20.0, augraphy_shift=99.0, dibco2017=14.0)
    write_eval(tmp_path, 'xd_dibco_unet', kaggle_test=18.0, augraphy_shift=15.0, dibco2017=99.0)
    mean = transfer_mean(build_matrix(tmp_path, 'psnr'))
    assert mean['unet'] == (16.0 + 12.0 + 20.0 + 14.0 + 18.0 + 15.0) / 6   # in-domain 30/99/99 ignored
    assert math.isnan(mean['cnn'])                                       # no eval JSON at all


def test_transfer_mean_with_a_missing_cell_is_nan(tmp_path):
    write_eval(tmp_path, 'unet_edge_aug', kaggle_test=30.0, augraphy_shift=16.0)   # no dibco2017
    write_eval(tmp_path, 'xd_augraphy_unet', kaggle_test=20.0, dibco2017=14.0)
    write_eval(tmp_path, 'xd_dibco_unet', kaggle_test=18.0, augraphy_shift=15.0)
    assert math.isnan(transfer_mean(build_matrix(tmp_path, 'psnr'))['unet'])


def test_transfer_mean_markdown_ranks_by_psnr(tmp_path):
    for model, v in (('unet_edge_aug', 10.0), ('xd_augraphy_unet', 10.0), ('xd_dibco_unet', 10.0),
                     ('nafnet_ft', 20.0), ('xd_augraphy_nafnet', 20.0), ('xd_dibco_nafnet', 20.0)):
        write_eval(tmp_path, model, kaggle_test=v, augraphy_shift=v, dibco2017=v)
    md = transfer_mean_markdown(tmp_path)
    assert md.index('NAFNet') < md.index('SmallUNet') and '| 1 |' in md and '20.00' in md
