"""Single inference path for every evaluation: full page or overlapping tiles."""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .evaluate import image_metrics
from .metrics_binary import dibco_metrics

IMAGE_SUFFIXES = {'.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff'}


def load_gray(path) -> np.ndarray:
    """Any common image mode -> float32 grayscale [0,1]; alpha composited on white."""
    with Image.open(path) as image:
        if image.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
            array = np.asarray(image, dtype=np.float32)
            peak = 65535.0 if image.mode.startswith('I;16') else max(float(array.max()), 1.0)
            return np.clip(array / peak, 0, 1).astype(np.float32)
        if image.mode == 'P' or image.mode in ('LA', 'PA'):
            image = image.convert('RGBA')
        if image.mode == 'RGBA':
            background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return np.asarray(image.convert('L'), dtype=np.float32) / 255.0


@torch.no_grad()
def _forward(model, img, device, amp):
    x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None].to(device)
    use_amp = amp and torch.device(device).type == 'cuda'
    with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
        y = model(x)
    return y.float()[0, 0].cpu().numpy()


def _ramp(length, overlap, blend_start, blend_end):
    """1-D tile weight. On sides shared with a neighbour tile the outer
    overlap/4 pixels get weight 0 (they suffer conv zero-padding artefacts),
    followed by a linear ramp over overlap/2 pixels. Image borders keep 1."""
    weights = np.ones(length, np.float32)
    margin, ramp = overlap // 4, max(overlap // 2, 1)
    edge = np.concatenate([np.zeros(margin, np.float32),
                           (np.arange(ramp, dtype=np.float32) + 1) / (ramp + 1)])
    if overlap > 0 and blend_start:
        weights[:edge.size] = edge
    if overlap > 0 and blend_end:
        weights[length - edge.size:] = np.minimum(weights[length - edge.size:], edge[::-1])
    return weights


def _starts(size, tile, step):
    starts = list(range(0, max(size - tile, 0) + 1, step))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


def denoise(model, img, device, tile=None, overlap=64, amp=False) -> np.ndarray:
    """Denoise a [H,W] page; output has the same H x W, clamped to [0,1]."""
    img = np.asarray(img, dtype=np.float32)
    height, width = img.shape
    model = model.to(device).eval()
    if tile is None or (height <= tile and width <= tile):
        return np.clip(_forward(model, img, device, amp), 0, 1)
    tile = int(tile)
    overlap = min(int(overlap), tile // 2)
    step = tile - overlap
    # Pad with white so every tile is full-size; crop at the end.
    pad_h, pad_w = max(0, tile - height), max(0, tile - width)
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), constant_values=1.0)
    ph, pw = padded.shape
    accum = np.zeros_like(padded)
    weight = np.zeros_like(padded)
    tops, lefts = _starts(ph, tile, step), _starts(pw, tile, step)
    for top in tops:
        for left in lefts:
            window = np.outer(_ramp(tile, overlap, top > 0, top + tile < ph),
                              _ramp(tile, overlap, left > 0, left + tile < pw))
            out = _forward(model, padded[top:top + tile, left:left + tile], device, amp)
            accum[top:top + tile, left:left + tile] += out * window
            weight[top:top + tile, left:left + tile] += window
    return np.clip(accum / weight, 0, 1)[:height, :width]


def _find_gt(gt_dir, stem):
    for candidate in (stem, f'{stem}_gt'):
        for path in Path(gt_dir).glob(f'{candidate}.*'):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                return path
    return None


def run_folder(model, in_dir, out_dir, device, gt_dir=None, tile=1024, amp=False) -> list:
    """Denoise every image in in_dir into out_dir/<stem>.png; score when GT exists.

    Binary GT -> DIBCO metrics on the Otsu-binarized output; otherwise PSNR/SSIM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(p for p in Path(in_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        img = load_gray(path)
        pred = denoise(model, img, device, tile=tile, amp=amp)
        Image.fromarray((pred * 255 + 0.5).astype(np.uint8)).save(out_dir / f'{path.stem}.png')
        row = {'id': path.stem, 'height': img.shape[0], 'width': img.shape[1]}
        gt_path = _find_gt(gt_dir, path.stem) if gt_dir is not None else None
        if gt_path is not None:
            gt = load_gray(gt_path)
            if gt.shape != pred.shape:
                raise ValueError(f'GT size mismatch for {path.name}')
            if np.isin(np.unique(gt), [0.0, 1.0]).all():
                row.update(dibco_metrics(pred, gt < 0.5))
            else:
                scores = image_metrics(torch.from_numpy(pred)[None, None], torch.from_numpy(gt)[None, None])
                row.update({k: float(v[0]) for k, v in scores.items()})
        rows.append(row)
    return rows


@torch.no_grad()
def measure_latency(model, shape=(1, 1, 540, 420), device='cpu', warmup=10, iters=50) -> dict:
    """Median / p90 forward latency (ms), peak CUDA memory (MB) and parameter count."""
    model = model.to(device).eval()
    x = torch.rand(*shape, device=device)
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    return {'median_ms': float(np.median(times)), 'p90_ms': float(np.percentile(times, 90)),
            'peak_vram_mb': float(torch.cuda.max_memory_allocated(device) / 2 ** 20) if cuda else 0.0,
            'params': int(sum(p.numel() for p in model.parameters()))}
