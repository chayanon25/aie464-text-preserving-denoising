"""Vendored NAFNet with grayscale wrapper."""
import pytest
import torch

from src.models import build_model
from src.models.nafnet import NAFNET_WEIGHTS, load_nafnet_sidd


@pytest.mark.parametrize('hw', [(37, 53), (64, 64)])
def test_nafnet_shape(hw):
    x = torch.rand(1, 1, *hw); assert build_model('nafnet').eval()(x).shape == x.shape


def test_nafnet_identity_at_init():
    # NAFNet blocks start with beta = gamma = 0 and the global residual; only the
    # ending conv perturbs the output, so zeroing it gives an exact identity.
    m = build_model('nafnet').eval()
    torch.nn.init.zeros_(m.net.ending.weight); torch.nn.init.zeros_(m.net.ending.bias)
    x = torch.rand(1, 1, 40, 56)
    assert torch.allclose(m(x), x, atol=1e-6)


def test_nafnet_backward_runs():
    m = build_model('nafnet'); x = torch.rand(1, 1, 32, 32)
    m(x).mean().backward()
    assert m.net.intro.weight.grad is not None


def test_nafnet_loads_sidd_if_present():
    if not NAFNET_WEIGHTS.exists(): pytest.skip('weights not downloaded')
    m = load_nafnet_sidd(NAFNET_WEIGHTS); assert sum(p.numel() for p in m.parameters()) > 1e6


def test_weights_env_read_at_call_time(monkeypatch, tmp_path):
    # Colab sets NAFNET_WEIGHTS after src is imported; the loader must still see it.
    missing = tmp_path / 'set_after_import.pth'
    monkeypatch.setenv('NAFNET_WEIGHTS', str(missing))
    with pytest.raises(FileNotFoundError, match='set_after_import'):
        load_nafnet_sidd()
