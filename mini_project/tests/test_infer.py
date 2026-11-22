"""Inference tests: tiled == full, odd sizes, input modes, folder runner."""
import numpy as np
import pytest
from PIL import Image

from src.infer import denoise, load_gray, measure_latency, run_folder
from src.models import build_model


def test_tiled_matches_full_for_cnn():
    m = build_model('cnn').eval(); img = np.random.rand(300, 517).astype(np.float32)
    assert np.abs(denoise(m, img, 'cpu', tile=128, overlap=32) - denoise(m, img, 'cpu')).max() < 1e-4


@pytest.mark.parametrize('hw', [(191, 245), (922, 1393), (1350, 2397)])
def test_tiled_shape_odd_large(hw):
    out = denoise(build_model('unet').eval(), np.random.rand(*hw).astype(np.float32), 'cpu', tile=512, overlap=64)
    assert out.shape == hw and 0 <= out.min() and out.max() <= 1


@pytest.mark.parametrize('mode', ['L', 'RGB', 'RGBA', 'P', 'I;16'])
def test_load_gray_modes(tmp_path, mode):
    p = tmp_path / f'x_{mode.replace(";", "")}.png'; Image.new(mode, (31, 17)).save(p)
    g = load_gray(p); assert g.shape == (17, 31) and g.dtype == np.float32 and 0 <= g.min() and g.max() <= 1


def test_load_gray_rgba_transparent_is_white(tmp_path):
    p = tmp_path / 't.png'; Image.new('RGBA', (8, 8), (0, 0, 0, 0)).save(p)
    assert load_gray(p).min() == pytest.approx(1.0)


def test_run_folder_with_binary_gt(tmp_path):
    (tmp_path / 'in').mkdir(); (tmp_path / 'gt').mkdir()
    Image.fromarray((np.random.rand(40, 50) * 255).astype(np.uint8)).save(tmp_path / 'in' / '1.png')
    Image.fromarray(np.full((40, 50), 255, np.uint8)).save(tmp_path / 'gt' / '1_gt.bmp')
    rows = run_folder(build_model('cnn').eval(), tmp_path / 'in', tmp_path / 'out', 'cpu', gt_dir=tmp_path / 'gt')
    assert (tmp_path / 'out' / '1.png').exists() and 'fm' in rows[0]


def test_measure_latency_keys():
    out = measure_latency(build_model('cnn'), shape=(1, 1, 32, 32), device='cpu', warmup=1, iters=3)
    assert set(out) == {'median_ms', 'p90_ms', 'peak_vram_mb', 'params'} and out['params'] == 7265
