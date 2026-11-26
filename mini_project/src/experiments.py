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
    JobConfig(job_id='nafnet_ft_dibco_lora8', data='dibco', init_from='nafnet_ft', lora_rank=8, lora_1x1=True,
              **{**_NAF, 'lr': 1e-3, 'min_lr': 1e-5}),
]

# Cross-domain benchmark (3 sources x 5 models, tested zero-shot on every test set).
# Each model keeps its Kaggle recipe; only `data` (and the KD teacher) changes.
CROSS_DOMAIN_MODELS = ('unet', 'dncnn', 'cnn', 'kd', 'nafnet')
_KAGGLE_REUSE = {'unet': 'unet_edge_aug', 'dncnn': 'dncnn_nobn_aug', 'kd': 'kd_cnn', 'nafnet': 'nafnet_ft'}


def cross_domain_job_id(source, model):
    if source == 'kaggle' and model in _KAGGLE_REUSE:
        return _KAGGLE_REUSE[model]
    return f'xd_{source}_{model}'


def _cross_domain_job(source, model):
    job_id = cross_domain_job_id(source, model)
    base = {**_IMPROVED, 'data': source}
    if model == 'unet':
        return JobConfig(job_id=job_id, model='unet', **base)
    if model == 'dncnn':
        return JobConfig(job_id=job_id, model='dncnn_nobn', **base)
    if model == 'cnn':   # no teacher: the baseline that KD is compared against
        return JobConfig(job_id=job_id, model='cnn', **base)
    if model == 'kd':    # teacher trained on the same source, never on Kaggle
        return JobConfig(job_id=job_id, model='cnn', kd_teacher=cross_domain_job_id(source, 'unet'),
                         kd_alpha=0.5, **base)
    return JobConfig(job_id=job_id, data=source, init_from='nafnet:sidd', **_NAF)


_jobs.append(_cross_domain_job('kaggle', 'cnn'))
_jobs += [_cross_domain_job(source, model) for source in ('augraphy', 'dibco') for model in CROSS_DOMAIN_MODELS]
JOBS = {job.job_id: job for job in _jobs}

# Same budgets as the Kaggle jobs they mirror; confirm after the pilot.
# Insertion order is run order: each KD teacher (unet) comes before its student.
CROSS_DOMAIN_BUDGETS = {'xd_kaggle_cnn': (30, 2000)}
for _source in ('augraphy', 'dibco'):
    for _model in CROSS_DOMAIN_MODELS:
        CROSS_DOMAIN_BUDGETS[cross_domain_job_id(_source, _model)] = (20, 1000) if _model == 'nafnet' else (30, 2000)


def with_budget(job_id, epochs, samples_per_epoch, **overrides):
    """Copy of a job with its training budget filled in (after the pilot)."""
    return replace(JOBS[job_id], epochs=int(epochs), samples_per_epoch=int(samples_per_epoch), **overrides)
