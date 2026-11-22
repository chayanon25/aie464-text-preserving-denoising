"""DIBCO-style binarization metrics on boolean ink masks (True = ink).

pFM and DRD are our own re-implementations of the published definitions
(pFM: Ntirogiannis et al. 2013 with skeleton-based pseudo-recall; DRD: Lu et
al. 2004). They have NOT been cross-checked against the official DIBCO
evaluation tool, so absolute values may differ slightly from published ones.
"""
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize


def otsu_binarize(img: np.ndarray) -> np.ndarray:
    """Grayscale [0,1] image (ink dark) -> boolean ink mask."""
    img = np.asarray(img, dtype=np.float32)
    if img.min() == img.max():
        return img < 0.5
    return img <= threshold_otsu(img)


def _f1(precision_hits, pred_count, recall_hits, gt_count):
    if pred_count == 0 or gt_count == 0 or precision_hits == 0 or recall_hits == 0:
        return 0.0
    precision = precision_hits / pred_count
    recall = recall_hits / gt_count
    return float(100 * 2 * precision * recall / (precision + recall))


def fmeasure(pred_ink, gt_ink) -> float:
    """Pixel F1 of ink, scaled to 0-100."""
    pred_ink, gt_ink = np.asarray(pred_ink, bool), np.asarray(gt_ink, bool)
    tp = int(np.logical_and(pred_ink, gt_ink).sum())
    return _f1(tp, int(pred_ink.sum()), tp, int(gt_ink.sum()))


def pseudo_fmeasure(pred_ink, gt_ink) -> float:
    """F1 with pseudo-recall measured against the GT skeleton (0-100)."""
    pred_ink, gt_ink = np.asarray(pred_ink, bool), np.asarray(gt_ink, bool)
    skeleton = skeletonize(gt_ink)
    tp = int(np.logical_and(pred_ink, gt_ink).sum())
    skel_hits = int(np.logical_and(pred_ink, skeleton).sum())
    return _f1(tp, int(pred_ink.sum()), skel_hits, int(skeleton.sum()))


def psnr_binary(pred_ink, gt_ink) -> float:
    """PSNR with C=1 on binary images; identical masks are capped at 100 dB."""
    mse = float(np.mean(np.asarray(pred_ink, bool) != np.asarray(gt_ink, bool)))
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def _drd_weights(size=5):
    half = size // 2
    i, j = np.mgrid[-half:half + 1, -half:half + 1]
    dist = np.sqrt(i ** 2 + j ** 2)
    weights = np.zeros_like(dist)
    weights[dist > 0] = 1.0 / dist[dist > 0]
    return weights / weights.sum()


def drd(pred_ink, gt_ink, block=8) -> float:
    """Distance-Reciprocal Distortion; neighbours outside the image are ignored."""
    pred_ink, gt_ink = np.asarray(pred_ink, bool), np.asarray(gt_ink, bool)
    weights = _drd_weights(5)
    height, width = gt_ink.shape
    padded = np.pad(gt_ink.astype(np.float64), 2)
    valid = np.pad(np.ones_like(gt_ink, dtype=np.float64), 2)
    total = 0.0
    for r, c in zip(*np.nonzero(pred_ink != gt_ink)):
        window = padded[r:r + 5, c:c + 5]
        inside = valid[r:r + 5, c:c + 5]
        total += float((np.abs(window - float(pred_ink[r, c])) * inside * weights).sum())
    blocks = gt_ink[:height // block * block, :width // block * block]
    blocks = blocks.reshape(height // block, block, width // block, block).swapaxes(1, 2)
    flat = blocks.reshape(blocks.shape[0], blocks.shape[1], -1)
    nubn = int(np.sum(flat.any(axis=-1) & ~flat.all(axis=-1)))
    return total / max(nubn, 1)


def dibco_metrics(pred_gray, gt_ink) -> dict:
    """Binarize a grayscale prediction with Otsu, then score against GT ink."""
    pred_ink = otsu_binarize(pred_gray)
    return {'fm': fmeasure(pred_ink, gt_ink), 'pfm': pseudo_fmeasure(pred_ink, gt_ink),
            'psnr': psnr_binary(pred_ink, gt_ink), 'drd': drd(pred_ink, gt_ink)}
