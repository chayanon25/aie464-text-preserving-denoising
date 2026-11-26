"""One shared pipeline for four models. Default: bounded smoke, never test."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from dataset import make_loader
from losses import DenoisingLoss
from models.shallow_cnn import ShallowCNN
from models.dncnn import DnCNNLite
from models.unet import SmallUNet
from benchmark import (EXPERIMENTS, MANIFEST_SHA256, digest, load_manifest, verify_data,
                       dataset_for, sync, memory, json_save, checkpoint_save,
                       evaluate_model, preview_ids, save_comparisons, save_loss)


@dataclass(frozen=True)
class TrainConfig:
    experiment: str
    mode: str = 'smoke'
    epochs: int = 1
    seed: int = 42
    patch_size: int = 256
    learning_rate: float = 1e-3
    edge_weight: float = 0.1

    def validate(self):
        if self.experiment not in EXPERIMENTS or self.mode not in ('smoke', 'train'):
            raise ValueError('Unknown experiment or mode')
        if self.seed != 42 or self.patch_size != 256:
            raise ValueError('Frozen seed=42 and patch_size=256 are required')
        if self.epochs < 1 or self.learning_rate <= 0 or self.edge_weight < 0:
            raise ValueError('Invalid training configuration')
        if self.mode == 'smoke' and self.epochs > 2:
            raise ValueError('Smoke is capped at two epochs, one optimizer update each')


def build_model(experiment):
    if experiment == 'cnn_l1':
        return ShallowCNN(width=16)
    if experiment == 'dncnn_lite':
        return DnCNNLite(width=32, depth=7)
    if experiment in ('unet_l1', 'unet_edge'):
        return SmallUNet(width=16)
    raise ValueError(f'Unknown experiment: {experiment}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def device_for(require_cuda):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('Choose a Colab T4 GPU runtime; no silent CPU full training')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def versions(device):
    import PIL
    return {'torch': str(torch.__version__), 'numpy': np.__version__, 'pillow': PIL.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}


def source_hashes(project_dir):
    files = ('dataset.py', 'evaluate.py', 'benchmark.py', 'train.py', 'losses.py',
             'models/shallow_cnn.py', 'models/dncnn.py', 'models/unet.py')
    return {name: digest(Path(project_dir) / 'src' / name) for name in files}


def aligned_crop(noisy, clean, size, generator):
    if noisy.shape != clean.shape or noisy.ndim != 4:
        raise ValueError('Crop requires matching paired [B,1,H,W] images')
    height, width = noisy.shape[-2:]
    if min(height, width) < size:
        raise ValueError('Page too small for the frozen crop size')
    top = int(torch.randint(height-size+1, (1,), generator=generator))
    left = int(torch.randint(width-size+1, (1,), generator=generator))
    crop = (..., slice(top, top+size), slice(left, left+size))
    return noisy[crop].contiguous(), clean[crop].contiguous(), [top, left, size, size]


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    name, values, position, gauss, cached = state['numpy']
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, gauss, cached))


def selected_checkpoint(last):
    """Best weights are for inference; only last.pt is an optimizer resume point."""
    keys = ('config', 'best_psnr', 'source_hashes', 'data_hashes', 'manifest_sha256',
            'versions', 'initial_state_sha256', 'test_evaluated')
    return {**{key: last[key] for key in keys}, 'model_state_dict': last['best_model_state_dict'],
            'completed_epochs': last['best_epoch'], 'inference_only': True}


def train_experiment(project_dir, output_dir, config, *, resume=False,
                     approved_training=False, require_cuda=True):
    """Smoke: one train update + two full validation pages per epoch.

    Training: all 100 training crops and 20 full validation pages each epoch.
    Resume from the last COMPLETED epoch; interrupted epoch is replayed.
    Never constructs, decodes or evaluates a test dataset.
    """
    config.validate()
    if config.mode == 'train' and not approved_training:
        raise PermissionError('Full training requires separate approval after pilot results')
    device = device_for(require_cuda)
    seed_everything(config.seed)
    project_dir, output_dir = Path(project_dir), Path(output_dir)
    manifest = load_manifest(project_dir)
    sources, data_hashes = source_hashes(project_dir), verify_data(project_dir, manifest)
    environment = versions(device)
    train_dataset = dataset_for(project_dir, manifest, 'train')
    val_ids = (preview_ids(manifest, 'val', 2) if config.mode == 'smoke' else manifest['splits']['val'])
    val_loader = make_loader(dataset_for(project_dir, manifest, 'val', val_ids))
    model = build_model(config.experiment).to(device)
    initial_hash = __import__('hashlib').sha256(b''.join(
        value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = DenoisingLoss(config.edge_weight if config.experiment == 'unet_edge' else 0).to(device)
    history, validation, completed, best_psnr = [], [], 0, -float('inf')
    best_weights, best_epoch = None, 0
    elapsed_before = 0.0
    if resume:
        saved = torch.load(output_dir / 'last.pt', map_location='cpu', weights_only=True)
        expected = (saved['config'] == asdict(config) and saved['source_hashes'] == sources and
                    saved['data_hashes'] == data_hashes and saved['manifest_sha256'] == MANIFEST_SHA256 and
                    saved['versions'] == environment)
        if not expected:
            raise ValueError('Resume rejected: config, source, data, split or environment changed')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scaler.load_state_dict(saved['scaler_state_dict'])
        restore_rng(saved['rng_state'])
        history, validation = saved['history'], saved['validation']
        completed, best_psnr = saved['completed_epochs'], saved['best_psnr']
        best_weights, best_epoch = saved['best_model_state_dict'], saved['best_epoch']
        elapsed_before = saved['run_seconds']
        # Repair a best.pt write interrupted after the last.pt epoch commit.
        if best_weights is not None:
            checkpoint_save(selected_checkpoint(saved), output_dir / 'best.pt')
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        # A disconnect during the first epoch can resume from the initial state.
        checkpoint_save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                         'scaler_state_dict': scaler.state_dict(), 'rng_state': rng_state(),
                         'config': asdict(config), 'completed_epochs': 0, 'best_psnr': best_psnr,
                         'best_model_state_dict': None, 'best_epoch': 0,
                         'history': [], 'validation': [], 'source_hashes': sources,
                         'data_hashes': data_hashes, 'manifest_sha256': MANIFEST_SHA256,
                         'versions': environment, 'initial_state_sha256': initial_hash,
                         'run_seconds': 0.0, 'test_evaluated': False}, output_dir / 'last.pt')
    json_save(asdict(config), output_dir / 'config.json')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    started = time.perf_counter()
    for epoch in range(completed, config.epochs):
        model.train()
        # Isolated generators guarantee the SAME page/crop sequence for all models,
        # independent of architecture, validation calls and previous epochs.
        loader = make_loader(train_dataset, shuffle=True, seed=config.seed + epoch)
        crop_generator = torch.Generator().manual_seed(config.seed + epoch)
        sync(device)
        train_started = time.perf_counter()
        for index, batch in enumerate(loader):
            if config.mode == 'smoke' and index == 1:
                break
            noisy, clean, coords = aligned_crop(batch['noisy'], batch['clean'], config.patch_size, crop_generator)
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                prediction = model(noisy)
            if prediction.shape != clean.shape:
                raise ValueError('Forward changed paired dimensions')
            losses = criterion(prediction, clean)
            if not torch.isfinite(losses['total']):
                raise RuntimeError('Non-finite loss')
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            sync(device)
            history.append({'step': len(history)+1, 'epoch': epoch+1, 'id': batch['id'][0], 'crop': coords,
                            **{key: value.item() for key, value in losses.items()},
                            'grad_norm': float(grad_norm), 'elapsed_seconds': elapsed_before + time.perf_counter()-started,
                            **memory(device)})
        train_seconds = time.perf_counter() - train_started
        metrics, _ = evaluate_model(model, val_loader, device, manifest)
        validation.append({'epoch': epoch+1, 'train_seconds': train_seconds, **metrics})
        improved = metrics['mean']['psnr'] > best_psnr
        if improved:
            best_psnr = metrics['mean']['psnr']
            best_epoch = epoch + 1
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                      'scaler_state_dict': scaler.state_dict(), 'rng_state': rng_state(),
                      'config': asdict(config), 'completed_epochs': epoch+1, 'best_psnr': best_psnr,
                      'best_model_state_dict': best_weights, 'best_epoch': best_epoch,
                      'history': history, 'validation': validation, 'source_hashes': sources,
                      'data_hashes': data_hashes, 'manifest_sha256': MANIFEST_SHA256,
                      'versions': environment, 'initial_state_sha256': initial_hash,
                      'run_seconds': elapsed_before + time.perf_counter()-started, 'test_evaluated': False}
        # One atomic last.pt commits both current and best weights. Resume can
        # reconstruct best.pt if interruption occurs between the two writes.
        checkpoint_save(checkpoint, output_dir / 'last.pt')
        checkpoint_save(selected_checkpoint(checkpoint), output_dir / 'best.pt')
        json_save(history, output_dir / 'step_log.json')
        json_save(validation, output_dir / 'validation.json')
        print(f"{config.experiment} epoch {epoch+1}/{config.epochs} | val PSNR {metrics['mean']['psnr']:.3f} | SSIM {metrics['mean']['ssim']:.4f}", flush=True)
    # Report and visualize the selected BEST validation checkpoint, not last weights.
    best = torch.load(output_dir / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    chosen = [p for p in preview_ids(manifest, 'val') if p in val_ids]
    metrics, samples = evaluate_model(model, val_loader, device, manifest, previews=chosen)
    summary = {'experiment': config.experiment, 'mode': config.mode, 'selected_epoch': best['completed_epochs'],
               'steps': len(history), 'config': asdict(config), 'manifest_sha256': MANIFEST_SHA256,
               'versions': environment, 'parameters': sum(p.numel() for p in model.parameters()),
               'run_seconds': elapsed_before + time.perf_counter()-started,
               'validation': metrics, 'gpu_memory': memory(device), 'test_evaluated': False}
    json_save(summary, output_dir / 'metrics.json')
    json_save(history, output_dir / 'step_log.json')
    json_save(validation, output_dir / 'validation.json')
    save_comparisons(samples, output_dir / 'comparisons.png')
    save_loss(history, output_dir / 'loss.png')
    # Reload proof for the best checkpoint; no test images involved.
    restored = build_model(config.experiment).to(device)
    restored.load_state_dict(best['model_state_dict'])
    model.eval()
    restored.eval()
    sample = next(iter(val_loader))['noisy'].to(device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        torch.testing.assert_close(model(sample), restored(sample), rtol=0, atol=0)
    return summary
