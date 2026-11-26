"""Evaluation suite: classical baselines, group summaries, midterm reproduction."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.classical import CLASSICAL
from src.data_external import load_dibco
from src.dataset import load_kaggle_split
from src.eval_suite import evaluate, load_midterm, predict_fn, save_eval

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'outputs' / 'phase1' / 'split_manifest.json'
# Benchmark JSON saved alongside the midterm checkpoints we actually have.
# results/final/ comes from a separate run whose unet_edge checkpoint is lost
# (31.865 vs 31.836 dB); the other three models are identical in both files.
BENCH_JSON = ROOT / 'backups' / 'aie464_final_artifacts' / 'final_test_benchmark.json'
RESULTS_JSON = ROOT / 'results' / 'final' / 'final_test_benchmark.json'


@pytest.mark.slow
@pytest.mark.parametrize('exp', ['cnn_l1', 'dncnn_lite', 'unet_l1', 'unet_edge'])
def test_reproduce_midterm_psnr(kaggle_root, exp):
    ref = json.loads(BENCH_JSON.read_text())[exp]['test_psnr']
    try:
        m = load_midterm(exp).eval()
    except FileNotFoundError:
        pytest.skip('midterm checkpoint not available')
    res = evaluate(predict_fn(m, 'cpu', tile=None), {'kaggle_test': load_kaggle_split(kaggle_root, MANIFEST, 'test')}, exp)
    assert res['summary']['kaggle_test']['psnr'] == pytest.approx(ref, abs=0.01)
    if exp != 'unet_edge':
        published = json.loads(RESULTS_JSON.read_text())[exp]['test_psnr']
        assert res['summary']['kaggle_test']['psnr'] == pytest.approx(published, abs=0.01)


def test_classical_shapes_and_range():
    img = np.random.rand(50, 70).astype(np.float32)
    for f in CLASSICAL.values():
        o = f(img); assert o.shape == img.shape and 0 <= o.min() and o.max() <= 1


def test_summary_has_group_breakdown(dibco_root):
    res = evaluate(CLASSICAL['otsu'], {'dibco2017': load_dibco(dibco_root, 2017)}, 'otsu')
    assert {'dibco2017', 'dibco2017/printed', 'dibco2017/handwritten'} <= set(res['summary'])
    assert {'fm', 'pfm', 'psnr', 'drd'} <= set(res['summary']['dibco2017'])


def test_gray_testset_metrics_and_save(tmp_path):
    clean = np.ones((32, 40), np.float32); clean[10:20, 10:20] = 0
    sample = {'id': 'a', 'noisy': clean * 0.9, 'clean': clean, 'group': 'g'}
    res = evaluate(CLASSICAL['noisy'], {'kaggle_test': [sample]}, 'noisy')
    assert {'psnr', 'ssim', 'text_psnr'} <= set(res['per_image'][0])
    save_eval(res, tmp_path / 'x.json')
    assert json.loads((tmp_path / 'x.json').read_text())['name'] == 'noisy'


def test_load_testsets_required_missing_raises(tmp_path):
    from src.eval_suite import load_testsets
    with pytest.raises(FileNotFoundError):
        load_testsets(tmp_path, MANIFEST, tmp_path, tmp_path, required=('kaggle_test', 'dibco2017'))


def test_gray_metrics_include_fm():
    from src.eval_suite import _gray_metrics
    from src.metrics_binary import fmeasure
    from src.smoke import make_smoke_samples
    clean = make_smoke_samples(1, seed=5)[0]['clean']
    m = _gray_metrics(clean, clean)
    assert m['fm'] == pytest.approx(fmeasure(clean < 0.5, clean < 0.5))
