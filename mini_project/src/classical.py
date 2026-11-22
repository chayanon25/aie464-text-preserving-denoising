"""Classical (non-learned) baselines; every function maps [H,W] [0,1] -> [H,W] [0,1]."""
import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu, threshold_sauvola


def identity(img):
    return np.clip(np.asarray(img, dtype=np.float32), 0, 1)


def median3(img):
    return np.clip(median_filter(np.asarray(img, dtype=np.float32), size=3), 0, 1)


def otsu(img):
    img = np.asarray(img, dtype=np.float32)
    if img.min() == img.max():
        return np.ones_like(img)
    return (img > threshold_otsu(img)).astype(np.float32)


def sauvola(img):
    img = np.asarray(img, dtype=np.float32)
    return (img > threshold_sauvola(img, window_size=25)).astype(np.float32)


CLASSICAL = {'noisy': identity, 'median3': median3, 'otsu': otsu, 'sauvola': sauvola}
