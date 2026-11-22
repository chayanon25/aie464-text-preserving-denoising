"""Full-page grayscale validation metrics, data range 1. No extra packages."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def image_metrics(prediction, target):
    """Return per-page PSNR (dB) and SSIM; inputs [B,1,H,W], clamped to [0,1].

    SSIM: Gaussian 11x11, sigma=1.5, valid windows, population covariance,
    K1=.01, K2=.03, no downsampling. Average per-page scores across validation.
    """
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError('Metrics require matching grayscale tensors [B,1,H,W]')
    if min(prediction.shape[-2:]) < 11:
        raise ValueError('SSIM requires both dimensions >= 11')
    x, y = prediction.float(), target.float()
    for tensor in (x, y):
        if not torch.isfinite(tensor).all():
            raise ValueError('Metric inputs must be finite')
    # Match the midterm notebook: clamp to [0,1] and cap identical pages at 100 dB.
    x, y = x.clamp(0, 1), y.clamp(0, 1)
    mse = (x - y).square().mean(dim=(1, 2, 3))
    psnr = -10 * torch.log10(mse.clamp_min(1e-10))
    coordinates = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    weights = torch.exp(-coordinates.square() / (2 * 1.5 ** 2))
    weights /= weights.sum()
    kernel = torch.outer(weights, weights)[None, None].expand(5, 1, 11, 11).contiguous()
    moments = F.conv2d(torch.cat([x, y, x.square(), y.square(), x * y], dim=1), kernel, groups=5)
    mx, my, ex2, ey2, exy = moments.split(1, dim=1)
    vx, vy = (ex2 - mx.square()).clamp_min(0), (ey2 - my.square()).clamp_min(0)
    covariance = exy - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mx * my + c1) * (2 * covariance + c2)
                / ((mx.square() + my.square() + c1) * (vx + vy + c2)))
    return {'psnr': psnr, 'ssim': ssim_map.mean(dim=(1, 2, 3))}


@torch.no_grad()
def text_masked_psnr(prediction, target, threshold=0.5):
    """Per-page PSNR over text pixels only (target < threshold); [B,1,H,W].

    A page without any text pixels yields NaN; no error on text yields +inf.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError('text_masked_psnr requires matching tensors [B,1,H,W]')
    x, y = prediction.float().clamp(0, 1), target.float().clamp(0, 1)
    mask = (y < threshold).float()
    count = mask.sum(dim=(1, 2, 3))
    mse = ((x - y).square() * mask).sum(dim=(1, 2, 3)) / count
    return -10 * torch.log10(mse)
