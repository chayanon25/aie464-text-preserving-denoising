"""Global (non-spatial) training augmentation for noisy/clean patch pairs.

Only a horizontal flip touches both images. Photometric, blur, JPEG and noise
effects apply to the noisy input only, so the clean target stays exact. No
spatially varying effects (gradients, bleed, folds) are used here: those are
reserved for the held-out Augraphy shift test set.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter


def _to_uint8(img):
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


class TrainAugment:
    def __init__(self, p_photometric=0.8, p_blur=0.3, p_jpeg=0.3, p_noise=0.3):
        self.p_photometric, self.p_blur = float(p_photometric), float(p_blur)
        self.p_jpeg, self.p_noise = float(p_jpeg), float(p_noise)

    def __call__(self, noisy, clean, rng):
        noisy = np.asarray(noisy, dtype=np.float32)
        clean = np.asarray(clean, dtype=np.float32)
        if rng.random() < 0.5:
            noisy, clean = noisy[:, ::-1], clean[:, ::-1]
        out = noisy.copy()
        if rng.random() < self.p_photometric:
            out = out + rng.uniform(-0.15, 0.15)
            out = (out - 0.5) * rng.uniform(0.7, 1.3) + 0.5
            out = np.clip(out, 0, 1) ** rng.uniform(0.7, 1.5)
        if rng.random() < self.p_blur:
            image = _to_uint8(out).filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
            out = np.asarray(image, dtype=np.float32) / 255.0
        if rng.random() < self.p_jpeg:
            buffer = BytesIO()
            _to_uint8(out).save(buffer, format='JPEG', quality=int(rng.integers(30, 96)))
            buffer.seek(0)
            with Image.open(buffer) as image:
                out = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
        if rng.random() < self.p_noise:
            out = out + rng.normal(0.0, rng.uniform(0.0, 0.03), size=out.shape)
        return np.clip(out, 0, 1).astype(np.float32), np.ascontiguousarray(clean)
