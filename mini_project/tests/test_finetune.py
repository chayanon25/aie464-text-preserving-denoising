"""Fine-tune techniques: LoRA for conv, encoder freezing, job definitions."""
import torch
from torch import nn

from src.experiments import JOBS
from src.lora import LoRAConv2d, apply_lora, freeze_encoder, merge_lora
from src.models import build_model
from src.train_engine import JobConfig, prepare_model


def test_lora_identity_at_init():
    m = build_model('unet').eval(); x = torch.rand(1, 1, 48, 64); ref = m(x)
    apply_lora(m, 4); assert torch.allclose(m(x), ref, atol=1e-6)


def test_lora_trainable_fraction():
    m = build_model('unet'); n = apply_lora(m, 4)
    assert 0 < n < 0.15 * 487009 and n == sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_lora_merge_equivalent():
    m = build_model('unet').eval(); apply_lora(m, 4)
    for mod in m.modules():
        if isinstance(mod, LoRAConv2d): nn.init.normal_(mod.B.weight, std=0.01)
    x = torch.rand(1, 1, 48, 64); y = m(x)
    merged = merge_lora(m)
    assert not any(isinstance(mod, LoRAConv2d) for mod in merged.modules())
    assert torch.allclose(merged(x), y, atol=1e-5)


def test_freeze_encoder():
    m = build_model('unet'); freeze_encoder(m)
    assert all(not p.requires_grad for mod in m.encoder_modules() for p in mod.parameters())
    assert all(p.requires_grad for p in m.dec1.parameters())


def test_jobs_finetune_use_dibco_and_init():
    for k in ('ft_full', 'ft_freeze_enc', 'ft_lora4', 'ft_lora8'):
        assert JOBS[k].data == 'dibco' and JOBS[k].init_from == 'unet_edge_aug'


def test_jobs_ids_complete_and_unfilled():
    expected = {'pilot_unet_edge_aug', 'unet_edge_aug', 'dncnn_nobn_aug', 'unet_aug_l0', 'unet_aug_l05',
                'ft_full', 'ft_freeze_enc', 'ft_lora4', 'ft_lora8', 'kd_cnn', 'unet_edge_aug_s43',
                'unet_edge_aug_s44', 'nafnet_short_ft', 'nafnet_ft', 'nafnet_ft_dibco_full', 'nafnet_ft_dibco_lora8'}
    original = {k for k in JOBS if not k.startswith('xd_')}   # xd_* = cross-domain jobs (test_experiments.py)
    assert original == expected and all(k == v.job_id for k, v in JOBS.items())


def test_prepare_model_lora_from_midterm():
    cfg = JobConfig(job_id='x', model='unet', init_from='midterm:unet_edge', lora_rank=4)
    try:
        m = prepare_model(cfg)
    except FileNotFoundError:
        return
    assert any(isinstance(mod, LoRAConv2d) for mod in m.modules())


def test_lora_1x1_reaches_nafnet_blocks():
    m = build_model('nafnet'); total = sum(p.numel() for p in m.parameters())
    n = apply_lora(m, 8, include_1x1=True)
    assert n > 0.01 * total
    m.eval(); x = torch.rand(1, 1, 32, 32)
    for mod in m.modules():
        if isinstance(mod, LoRAConv2d): nn.init.normal_(mod.B.weight, std=0.01)
    y = m(x); assert torch.allclose(merge_lora(m)(x), y, atol=1e-4)


def test_nafnet_lora_job_uses_1x1():
    assert JOBS['nafnet_ft_dibco_lora8'].lora_1x1 is True
