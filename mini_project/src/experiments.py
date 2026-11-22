"""Job definitions (spec 6.3). epochs / samples_per_epoch stay None until the
pilot (Task 12) measures throughput; run_job refuses to start without them.

Hyperparameter choices left to the builder (spec section 8):
- fine-tune lr 1e-4 (full / freeze encoder), 1e-3 for LoRA (few, zero-init params)
- KD alpha 0.5 (teacher = unet_edge_aug, student = ShallowCNN)
- NAFNet batch 4 to stay well inside 14 GB on a T4 with AMP
"""
from dataclasses import replace

from .train_engine import JobConfig

_IMPROVED = dict(data='kaggle', edge_weight=0.1, augment=True, scheduler='cosine', amp=True)
_FT = dict(data='dibco', edge_weight=0.1, augment=True, scheduler='cosine', amp=True, init_from='unet_edge_aug')
_NAF = dict(model='nafnet', batch_size=4, lr=1e-4, min_lr=1e-6, amp=True, augment=True, edge_weight=0.1)

_jobs = [
    JobConfig(job_id='pilot_unet_edge_aug', model='unet', **_IMPROVED),
    JobConfig(job_id='unet_edge_aug', model='unet', **_IMPROVED),
    JobConfig(job_id='dncnn_nobn_aug', model='dncnn_nobn', **_IMPROVED),
    JobConfig(job_id='unet_aug_l0', model='unet', **{**_IMPROVED, 'edge_weight': 0.0}),
    JobConfig(job_id='unet_aug_l05', model='unet', **{**_IMPROVED, 'edge_weight': 0.5}),
    JobConfig(job_id='ft_full', model='unet', lr=1e-4, min_lr=1e-6, **_FT),
    JobConfig(job_id='ft_freeze_enc', model='unet', lr=1e-4, min_lr=1e-6, freeze_encoder=True, **_FT),
    JobConfig(job_id='ft_lora4', model='unet', lr=1e-3, min_lr=1e-5, lora_rank=4, **_FT),
    JobConfig(job_id='ft_lora8', model='unet', lr=1e-3, min_lr=1e-5, lora_rank=8, **_FT),
    JobConfig(job_id='kd_cnn', model='cnn', kd_teacher='unet_edge_aug', kd_alpha=0.5, **_IMPROVED),
    JobConfig(job_id='unet_edge_aug_s43', model='unet', seed=43, **_IMPROVED),
    JobConfig(job_id='unet_edge_aug_s44', model='unet', seed=44, **_IMPROVED),
    JobConfig(job_id='nafnet_short_ft', data='kaggle', init_from='nafnet:sidd', **_NAF),
    JobConfig(job_id='nafnet_ft', data='kaggle', init_from='nafnet:sidd', **_NAF),
    JobConfig(job_id='nafnet_ft_dibco_full', data='dibco', init_from='nafnet_ft', **_NAF),
    JobConfig(job_id='nafnet_ft_dibco_lora8', data='dibco', init_from='nafnet_ft', lora_rank=8,
              **{**_NAF, 'lr': 1e-3, 'min_lr': 1e-5}),
]
JOBS = {job.job_id: job for job in _jobs}


def with_budget(job_id, epochs, samples_per_epoch, **overrides):
    """Copy of a job with its training budget filled in (after the pilot)."""
    return replace(JOBS[job_id], epochs=int(epochs), samples_per_epoch=int(samples_per_epoch), **overrides)
