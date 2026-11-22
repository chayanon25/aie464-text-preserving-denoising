"""Model architecture tests: parameter counts, shapes, midterm checkpoints."""
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models import MIDTERM_EXPERIMENTS, build_model

MIDTERM_CKPT_DIR = (Path(__file__).resolve().parents[1]
                    / 'backups' / 'aie464_final_artifacts' / 'checkpoints')


@pytest.mark.parametrize('name,count', [('cnn', 7265), ('dncnn_bn', 47009),
                                        ('unet', 487009), ('dncnn_nobn', 46849)])
def test_param_counts(name, count):
    assert sum(p.numel() for p in build_model(name).parameters()) == count


@pytest.mark.parametrize('name', ['cnn', 'dncnn_bn', 'dncnn_nobn', 'unet'])
@pytest.mark.parametrize('hw', [(37, 53), (64, 64), (191, 245)])
def test_shape_preserved(name, hw):
    x = torch.rand(1, 1, *hw)
    assert build_model(name).eval()(x).shape == x.shape


def test_unet_is_residual():
    m = build_model('unet').eval()
    nn.init.zeros_(m.head.weight)
    nn.init.zeros_(m.head.bias)
    x = torch.rand(1, 1, 40, 56)
    assert torch.equal(m(x), x)


def test_dncnn_nobn_has_no_batchnorm():
    assert not any(isinstance(mod, nn.BatchNorm2d)
                   for mod in build_model('dncnn_nobn').modules())


def test_unet_encoder_modules():
    m = build_model('unet')
    assert m.encoder_modules() == [m.enc1, m.enc2, m.enc3, m.bridge]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model('resnet')


@pytest.mark.parametrize('exp', ['cnn_l1', 'dncnn_lite', 'unet_l1', 'unet_edge'])
def test_midterm_checkpoint_loads_strict(exp):
    ckpt = MIDTERM_CKPT_DIR / exp / 'best.pt'
    if not ckpt.exists():
        pytest.skip('midterm checkpoint not available')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)['model_state_dict']
    build_model(MIDTERM_EXPERIMENTS[exp]).load_state_dict(state, strict=True)
