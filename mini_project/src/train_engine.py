"""Resumable training engine: one job = one run directory on Drive.

Every epoch writes `last.pt` (model, optimizer, scheduler, scaler, all RNG
states, epoch, best metric, history) atomically via `*.tmp` + os.replace, so a
Colab disconnect at any moment leaves either the previous or the new
checkpoint, never a corrupt one. Re-running the same job resumes from
`last.pt`. Bit-exact resume is guaranteed on CPU; on GPU cuDNN kernels may
introduce tiny nondeterminism.
"""
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .augment import TrainAugment
from .dataset import RandomPatchDataset
from .evaluate import image_metrics
from .infer import denoise
from .losses import DenoisingLoss
from .models import build_model

DIBCO_VAL_TILE = 1024


@dataclass
class JobConfig:
    job_id: str
    model: str
    data: str = 'kaggle'                 # 'kaggle' | 'dibco'
    epochs: int | None = None            # filled in after the pilot (Task 12)
    samples_per_epoch: int | None = None
    edge_weight: float = 0.1
    batch_size: int = 8
    patch: int = 256
    lr: float = 1e-3
    min_lr: float = 1e-5
    scheduler: str = 'cosine'
    seed: int = 42
    amp: bool = True
    augment: bool = True
    init_from: str | None = None         # job_id (runs_root/<id>/best.pt) or 'midterm:<exp>'
    freeze_encoder: bool = False
    lora_rank: int | None = None
    lora_1x1: bool = False               # also wrap 1x1 convs (needed for NAFNet)
    git_sha: str | None = None
    kd_teacher: str | None = None        # job_id, run dir, checkpoint path or 'midterm:<exp>'
    kd_alpha: float = 0.5
    grad_clip: float = 1.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def _resolve_checkpoint(ref, runs_root):
    """Map a job_id / directory / file reference to a checkpoint path."""
    path = Path(ref)
    if path.is_file():
        return path
    if path.is_dir():
        return path / 'best.pt'
    if runs_root is not None:
        return Path(runs_root) / ref / 'best.pt'
    raise FileNotFoundError(f'Cannot resolve checkpoint reference: {ref}')


def load_trained(ref, runs_root=None, device='cpu'):
    """Load a model saved by run_job (rebuilding LoRA wrappers when needed),
    or a midterm checkpoint via 'midterm:<exp>'. Returns a model in eval mode."""
    if str(ref).startswith('midterm:'):
        from .eval_suite import load_midterm
        return load_midterm(str(ref).split(':', 1)[1]).to(device).eval()
    if str(ref) == 'nafnet:sidd':
        from .models.nafnet import load_nafnet_sidd
        return load_nafnet_sidd().to(device).eval()
    ckpt = torch.load(_resolve_checkpoint(ref, runs_root), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model = build_model(cfg['model'])
    if cfg.get('lora_rank'):
        from .lora import apply_lora
        apply_lora(model, cfg['lora_rank'], include_1x1=cfg.get('lora_1x1', False))
    model.load_state_dict(ckpt['model'], strict=True)
    return model.to(device).eval()


def prepare_model(cfg, runs_root=None):
    """Build -> load init_from -> freeze encoder -> wrap with LoRA."""
    model = build_model(cfg.model)
    if cfg.init_from:
        source = load_trained(cfg.init_from, runs_root)
        model.load_state_dict(source.state_dict(), strict=True)
    if cfg.freeze_encoder:
        from .lora import freeze_encoder
        freeze_encoder(model)
    if cfg.lora_rank:
        from .lora import apply_lora
        apply_lora(model, cfg.lora_rank, include_1x1=cfg.lora_1x1)
    return model


def _rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    # Checkpoints load with map_location=device, so the RNG tensors may sit on the GPU.
    torch.set_rng_state(state['torch'].cpu().to(torch.uint8))
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in state['cuda']])


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validate(model, samples, data_kind, device, amp=False):
    """Mean full-page PSNR on the validation pages (DIBCO pages are tiled)."""
    model.eval()
    tile = DIBCO_VAL_TILE if data_kind == 'dibco' else None
    scores = []
    for sample in samples:
        pred = denoise(model, sample['noisy'], device, tile=tile, amp=amp)
        metrics = image_metrics(torch.from_numpy(pred)[None, None], torch.from_numpy(sample['clean'])[None, None])
        scores.append(float(metrics['psnr'][0]))
    model.train()
    return float(np.mean(scores))


def run_job(cfg, run_dir, data, device, max_minutes=None, max_epochs=None, runs_root=None,
            on_progress=None, progress_every=50):
    """Train (or resume) one job. data = {'train': samples, 'val': samples}.

    Stops early with status 'paused' after `max_minutes` of wall time or after
    `max_epochs` epochs in this call; calling again resumes from last.pt.
    `on_progress()` is called every `progress_every` optimizer steps (registry heartbeat).
    """
    if not cfg.epochs or not cfg.samples_per_epoch:
        raise ValueError(f'{cfg.job_id}: epochs and samples_per_epoch must be set before training')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    runs_root = run_dir.parent if runs_root is None else Path(runs_root)
    device = torch.device(device)
    cuda = device.type == 'cuda'
    use_amp = bool(cfg.amp and cuda)

    _seed_everything(cfg.seed)
    model = prepare_model(cfg, runs_root).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr)
    steps_per_epoch = math.ceil(cfg.samples_per_epoch / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=cfg.min_lr)
    elif cfg.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    else:
        raise ValueError(f'Unknown scheduler: {cfg.scheduler}')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    criterion = DenoisingLoss(edge_weight=cfg.edge_weight).to(device)
    teacher = load_trained(cfg.kd_teacher, runs_root, device) if cfg.kd_teacher else None
    if teacher is not None:
        for p in teacher.parameters():
            p.requires_grad_(False)

    start_epoch, best, history = 0, -math.inf, []
    last_path = run_dir / 'last.pt'
    if last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        _set_rng_state(ckpt['rng'])
        start_epoch, best, history = ckpt['epoch'], ckpt['best'], ckpt['history']

    augment = TrainAugment() if cfg.augment else None
    dataset = RandomPatchDataset(data['train'], cfg.patch, cfg.samples_per_epoch, augment=augment, seed=cfg.seed)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)
    started, epochs_this_call, steps_this_call = time.time(), 0, 0
    model.train()
    for epoch in range(start_epoch, cfg.epochs):
        if max_epochs is not None and epochs_this_call >= max_epochs:
            break
        if max_minutes is not None and (time.time() - started) / 60 >= max_minutes:
            break
        dataset.set_epoch(epoch)
        epoch_start, loss_sum, count = time.time(), 0.0, 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(noisy)
            loss = criterion(pred, clean)['total']
            if teacher is not None and cfg.kd_alpha > 0:
                with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
                    target = teacher(noisy)
                loss = loss + cfg.kd_alpha * F.l1_loss(pred.float(), target.float())
            if not torch.isfinite(loss):
                raise RuntimeError(f'{cfg.job_id}: non-finite loss at epoch {epoch + 1}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * noisy.shape[0]
            count += noisy.shape[0]
            steps_this_call += 1
            if on_progress is not None and steps_this_call % progress_every == 0:
                on_progress()
        val_psnr = validate(model, data['val'], cfg.data, device, amp=use_amp)
        record = {'epoch': epoch + 1, 'train_loss': loss_sum / max(count, 1), 'val_psnr': val_psnr,
                  'lr': optimizer.param_groups[0]['lr'], 'seconds': time.time() - epoch_start}
        if cuda:
            record['peak_vram_mb'] = torch.cuda.max_memory_allocated(device) / 2 ** 20
        history.append(record)
        if val_psnr > best:
            best = val_psnr
            atomic_save({'model': model.state_dict(), 'epoch': epoch + 1, 'val_psnr': val_psnr,
                         'config': cfg.to_dict(), 'git_sha': cfg.git_sha}, run_dir / 'best.pt')
        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': _rng_state(),
                     'epoch': epoch + 1, 'best': best, 'history': history,
                     'config': cfg.to_dict(), 'git_sha': cfg.git_sha}, last_path)
        atomic_write_text(run_dir / 'validation.json', json.dumps(
            {'job_id': cfg.job_id, 'git_sha': cfg.git_sha, 'config': cfg.to_dict(),
             'best_val_psnr': best, 'history': history}, indent=2) + '\n')
        epochs_this_call += 1
    status = 'done' if len(history) >= cfg.epochs else 'paused'
    return {'status': status, 'job_id': cfg.job_id, 'best_val_psnr': best, 'history': history}
