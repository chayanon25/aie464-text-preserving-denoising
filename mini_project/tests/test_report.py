"""Report tables and figure helpers (Task 13)."""
import json
import sys
from pathlib import Path

from src.report import build_tables

MINI_PROJECT = Path(__file__).resolve().parents[1]


def write_fake_evals(root):
    rows = {'a': 1.0, 'midterm_unet_edge': 0.0}
    for name, bump in rows.items():
        per = []
        for i in range(6):
            per.append({'testset': 'kaggle_test', 'id': f'{i}.png', 'group': f'p{i % 2}',
                        'psnr': 30 + bump + i * 0.1, 'ssim': 0.9, 'text_psnr': 25 + bump})
            per.append({'testset': 'augraphy_shift', 'id': f'{i}.png', 'group': f'p{i % 2}',
                        'psnr': 15 + bump + i * 0.1, 'ssim': 0.8, 'text_psnr': 9 + bump})
            per.append({'testset': 'dibco2017', 'id': f'2017/{i}', 'group': 'printed' if i % 2 else 'handwritten',
                        'fm': 80 + bump + i, 'pfm': 85, 'psnr': 14, 'drd': 6})
        (root / f'{name}.json').write_text(json.dumps({'name': name, 'per_image': per}))
    (root / 'latency.json').write_text(json.dumps({n: {'median_ms': 9.0, 'p90_ms': 10.0, 'peak_vram_mb': 100.0,
                                                      'params': 1000} for n in rows}))


def test_tables_from_fixture(tmp_path):
    write_fake_evals(tmp_path)
    t = build_tables(tmp_path, latency_path=tmp_path / 'latency.json',
                     pairs=[('a', 'midterm_unet_edge', 'kaggle_test', 'psnr')])
    assert {'in_domain', 'augraphy_shift', 'dibco2017', 'efficiency', 'wilcoxon'} <= set(t)
    assert '| model |' in t['in_domain'].lower() and 'printed' in t['dibco2017']
    assert '| a |' in t['wilcoxon']


def test_unet_diagram_shapes_from_code():
    sys.path.insert(0, str(MINI_PROJECT / 'tools'))
    from make_figures import unet_block_shapes
    shapes = unet_block_shapes(256)
    assert shapes['enc1'] == (16, 256, 256) and shapes['bridge'] == (128, 32, 32)
