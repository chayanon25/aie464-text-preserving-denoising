"""Grayscale, text-masked and DIBCO binary metric tests."""
import numpy as np
import pytest
import torch

from src.evaluate import image_metrics, text_masked_psnr
from src.metrics_binary import dibco_metrics, drd, fmeasure, otsu_binarize, psnr_binary
from src.stats import paired_wilcoxon


def test_text_masked_psnr_ignores_background():
    t = torch.ones(1, 1, 16, 16); t[..., 4:8, 4:8] = 0
    p = t.clone(); p[..., 12:, 12:] = 0.5   # error only in background
    assert torch.isinf(text_masked_psnr(p, t)).all()


def test_text_masked_psnr_counts_ink_error():
    t = torch.ones(1, 1, 16, 16); t[..., 4:8, 4:8] = 0
    p = t.clone(); p[..., 4:8, 4:8] = 0.1
    assert torch.allclose(text_masked_psnr(p, t), torch.tensor([20.0]))


def test_image_metrics_identical_is_finite_and_clamps():
    x = torch.rand(1, 1, 32, 32)
    out = image_metrics(x + 2.0, torch.ones_like(x))
    assert torch.allclose(out['psnr'], torch.tensor([100.0]))


def test_binary_perfect():
    gt = np.zeros((32, 32), bool); gt[8:24, 10:12] = True
    m = dibco_metrics(np.where(gt, 0.0, 1.0), gt)
    assert m['fm'] == pytest.approx(100) and m['pfm'] == pytest.approx(100) and m['drd'] == 0


def test_binary_all_background_pred():
    gt = np.zeros((32, 32), bool); gt[8:24, 10:12] = True
    assert fmeasure(np.zeros_like(gt), gt) == 0


def test_drd_single_isolated_flip():
    gt = np.zeros((16, 16), bool); gt[0, 0] = True        # exactly one non-uniform 8x8 block
    pred = gt.copy(); pred[11, 11] = True                  # isolated flip in a white area
    assert drd(pred, gt) == pytest.approx(1.0)


def test_psnr_binary_one_flip():
    gt = np.zeros((10, 10), bool); pred = gt.copy(); pred[0, 0] = True
    assert psnr_binary(pred, gt) == pytest.approx(20.0)


def test_otsu_ink_is_dark():
    img = np.ones((20, 20), np.float32); img[5:10, 5:10] = 0.1
    assert otsu_binarize(img)[7, 7] and not otsu_binarize(img)[0, 0]


def test_paired_wilcoxon_keys():
    out = paired_wilcoxon([1.0, 2.0, 3.0, 4.0, 5.0], [0.5, 1.0, 2.9, 3.0, 4.1])
    assert set(out) == {'statistic', 'pvalue', 'n', 'median_diff'} and out['n'] == 5
    assert out['median_diff'] == pytest.approx(0.9)
