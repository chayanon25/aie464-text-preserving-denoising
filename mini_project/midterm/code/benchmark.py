"""Frozen split checks, shared validation, artifacts and gated final benchmark."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import struct
import time

import numpy as np
from PIL import Image
import torch

from dataset import PairedDocumentDataset, make_loader
from evaluate import image_metrics

MANIFEST_SHA256 = '5b569d65a4d21aacff84e25986fce522847ce61a4575cd720cd60ca4a4180539'
EXPERIMENTS = ('cnn_l1', 'dncnn_lite', 'unet_l1', 'unet_edge')
MEMORY_LIMIT = 6 * 1024**3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(project_dir):
    path = Path(project_dir) / 'outputs/phase1/split_manifest.json'
    if digest(path) != MANIFEST_SHA256:
        raise ValueError('Frozen split manifest checksum differs; no new split is allowed')
    manifest = json.loads(path.read_text())
    for name, count, groups in (('train', 100, 25), ('val', 20, 5), ('test', 24, 6)):
        ids, gids = manifest['splits'][name], manifest['group_splits'][name]
        if len(set(ids)) != count or len(ids) != count or len(set(gids)) != groups:
            raise ValueError(f'Invalid split counts: {name}')
        if {p for g in gids for p in manifest['groups'][g]} != set(ids):
            raise ValueError('Page/group assignments differ')
    for left, right in itertools.combinations(('train', 'val', 'test'), 2):
        if (set(manifest['splits'][left]) & set(manifest['splits'][right]) or
                set(manifest['group_splits'][left]) & set(manifest['group_splits'][right])):
            raise ValueError('Group or page overlap')
    return manifest


def group_map(manifest):
    return {page: group for group, pages in manifest['groups'].items() for page in pages}


def verify_data(project_dir, manifest, splits=('train', 'val')):
    """Decode only explicitly requested splits; default never opens test images."""
    root, fingerprints = Path(project_dir) / 'data/raw', {}
    mapping = group_map(manifest)
    for split in splits:
        for page in manifest['splits'][split]:
            dimensions = []
            for folder in ('train', 'train_cleaned'):
                path = root / folder / page
                with Image.open(path) as image:
                    pixels = np.array(image.convert('L'), dtype=np.uint8)
                height, width = pixels.shape
                dimensions.append(pixels.shape)
                if folder == 'train_cleaned':
                    actual = hashlib.sha256(struct.pack('>II', width, height) + pixels.tobytes()).hexdigest()
                    if actual != mapping[page]:
                        raise ValueError(f'Clean page hash mismatch: {page}')
                fingerprints[f'{folder}/{page}'] = digest(path)
            if dimensions[0] != dimensions[1] or min(dimensions[0]) < 11:
                raise ValueError(f'Invalid paired dimensions: {page}')
    return fingerprints


def dataset_for(project_dir, manifest, split, ids=None):
    root = Path(project_dir) / 'data/raw'
    return PairedDocumentDataset(root / 'train', root / 'train_cleaned',
                                 manifest['splits'][split] if ids is None else ids)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def memory(device):
    result = {'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
    if device.type == 'cuda':
        result = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        if max(result.values()) > MEMORY_LIMIT:
            raise RuntimeError('PyTorch GPU memory exceeded 6 GiB; stop and review the pilot')
    return result


def json_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checkpoint_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def preview_ids(manifest, split, limit=3):
    return [manifest['groups'][g][0] for g in manifest['group_splits'][split]][:limit]


@torch.inference_mode()
def evaluate_model(model, loader, device, manifest, previews=(), repeats=1):
    """Full pages, float32 PSNR/SSIM; synchronized forward-only latency.

    Warm up on this loader (validation during development). Include model
    padding and output clamp, exclude disk IO, transfers and metric kernels.
    """
    model.eval()
    amp = device.type == 'cuda'
    first = next(iter(loader))['noisy'].to(device)
    for _ in range(2):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            model(first).float().clamp(0, 1)
    sync(device)
    started = time.perf_counter()
    rows, samples = [], []
    mapping = group_map(manifest)
    for batch in loader:
        noisy, clean = batch['noisy'].to(device), batch['clean'].to(device)
        timings = []
        for _ in range(repeats):
            sync(device)
            tick = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                denoised = model(noisy).float().clamp(0, 1)
            sync(device)
            timings.append(1000 * (time.perf_counter() - tick))
        if denoised.shape != clean.shape:
            raise ValueError('Model changed spatial dimensions')
        scores, baseline = image_metrics(denoised, clean), image_metrics(noisy, clean)
        row = {'id': batch['id'][0], 'group': mapping[batch['id'][0]],
               'psnr': scores['psnr'].item(), 'ssim': scores['ssim'].item(),
               'noisy_psnr': baseline['psnr'].item(), 'noisy_ssim': baseline['ssim'].item(),
               'l1': torch.nn.functional.l1_loss(denoised, clean).item(),
               'inference_ms': float(np.median(timings))}
        if not all(np.isfinite(v) for k, v in row.items() if k not in ('id', 'group')):
            raise RuntimeError('Non-finite metrics; inspect results rather than silently averaging')
        rows.append(row)
        memory(device)
        if row['id'] in previews:
            samples.append({'id': row['id'], 'noisy': noisy[0, 0].cpu(),
                            'denoised': denoised[0, 0].cpu(), 'clean': clean[0, 0].cpu()})
    sync(device)
    keys = ('psnr', 'ssim', 'noisy_psnr', 'noisy_ssim', 'l1', 'inference_ms')
    mean = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    groups = {g: {key: float(np.mean([r[key] for r in rows if r['group'] == g])) for key in keys}
              for g in sorted({r['group'] for r in rows})}
    return {'rows': rows, 'mean': mean, 'group_means': groups,
            'seconds': time.perf_counter() - started, 'gpu_memory': memory(device)}, samples


def save_comparisons(samples, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(samples) * 2, 3, figsize=(12, 5 * len(samples)), squeeze=False)
    for i, sample in enumerate(samples):
        height, width = sample['clean'].shape
        size = min(128, height, width)
        top, left = (height-size)//2, (width-size)//2
        for j, key in enumerate(('noisy', 'denoised', 'clean')):
            for row, pixels in ((2*i, sample[key]), (2*i+1, sample[key][top:top+size, left:left+size])):
                axes[row, j].imshow(pixels.numpy(), cmap='gray', vmin=0, vmax=1)
                axes[row, j].set_title(f"{sample['id']} | {key} | {'page' if row % 2 == 0 else 'fixed center crop'}")
                axes[row, j].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_loss(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    for key in ('total', 'l1', 'edge'):
        ax.plot([r['step'] for r in history], [r[key] for r in history], label=key)
    ax.set(xlabel='Optimizer update', ylabel='Loss')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def verify_selected_checkpoint(last, best):
    """Reject a stale best.pt before any final test pixels can be opened."""
    expected = last['best_model_state_dict']
    actual = best['model_state_dict']
    if (expected is None or best['completed_epochs'] != last['best_epoch'] or
            set(expected) != set(actual) or
            any(not torch.equal(expected[key].cpu(), actual[key].cpu()) for key in expected)):
        raise ValueError('best.pt is stale; resume the training run to repair it before final evaluation')


def prepare_final_staging(run_dir, protocol):
    """Resume-safe artifact staging; independently testable without test data."""
    staging = Path(run_dir) / 'final_test.partial'
    if staging.exists():
        marker = staging / 'protocol.json'
        contents = {path.name for path in staging.iterdir()}
        if not marker.exists() and contents.issubset({'protocol.json.tmp'}):
            # No model results exist yet: recover mkdir or an interrupted marker
            # write. Even a truncated .tmp can safely be replaced here.
            json_save(protocol, marker)
        elif not marker.is_file() or json.loads(marker.read_text()) != protocol:
            raise ValueError('Partial final benchmark protocol changed; refusing mixed results')
    else:
        staging.mkdir()
        json_save(protocol, staging / 'protocol.json')
    return staging


def final_benchmark(project_dir, run_dir, *, approved=False, require_cuda=True):
    """Explicit final-only gate. Never called by training or smoke tests."""
    if not approved:
        raise PermissionError('Final test evaluation requires separate user approval')
    from train import build_model, seed_everything, source_hashes, device_for, versions
    device = device_for(require_cuda)
    seed_everything(42)
    run_dir = Path(run_dir)
    destination = run_dir / 'final_test'
    if destination.exists():
        raise FileExistsError('Final test results already exist; refusing to overwrite')
    manifest = load_manifest(project_dir)
    validation_hashes = verify_data(project_dir, manifest)
    current_sources = source_hashes(project_dir)
    checkpoints, configs = {}, []
    for experiment in EXPERIMENTS:
        folder = run_dir / experiment
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
        verify_selected_checkpoint(last, best)
        config = last['config']
        if (config['mode'] != 'train' or config['experiment'] != experiment or
                last['completed_epochs'] != config['epochs']):
            raise ValueError('All four complete training runs are required before final evaluation')
        if (len(last['history']) != 100 * config['epochs'] or len(last['validation']) != config['epochs'] or
                any(len(epoch['rows']) != 20 or {row['id'] for row in epoch['rows']} != set(manifest['splits']['val'])
                    for epoch in last['validation'])):
            raise ValueError('Incomplete training or validation coverage')
        for checkpoint in (last, best):
            if (checkpoint['source_hashes'] != current_sources or
                    checkpoint['data_hashes'] != validation_hashes or
                    checkpoint['manifest_sha256'] != MANIFEST_SHA256 or checkpoint['config'] != config):
                raise ValueError('Checkpoint provenance differs from frozen development protocol')
        if last['versions'] != best['versions'] or last['versions'] != __import__('train').versions(device):
            raise ValueError('Use the same software/hardware environment for final benchmarking')
        common = {k: v for k, v in config.items() if k != 'experiment'}
        configs.append(common)
        checkpoints[experiment] = best
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError('Training budgets or protocols differ across models')
    # Only now, after all gates, decode the held-out test images.
    test_hashes = verify_data(project_dir, manifest, splits=('test',))
    loader = make_loader(dataset_for(project_dir, manifest, 'test'))
    # Publish only a complete four-model benchmark. Interrupted evaluation may
    # resume with exactly the same frozen protocol and checkpoint fingerprints.
    protocol = {'manifest_sha256': MANIFEST_SHA256, 'config': configs[0],
                'source_hashes': current_sources, 'test_data_hashes': test_hashes,
                'checkpoint_hashes': {name: digest(run_dir / name / 'best.pt') for name in EXPERIMENTS},
                'versions': versions(device)}
    staging = prepare_final_staging(run_dir, protocol)
    results, summary = {}, []
    for experiment, checkpoint in checkpoints.items():
        completed_file = staging / f'{experiment}.json'
        if completed_file.exists():
            metrics = json.loads(completed_file.read_text())
            if (not (staging / f'{experiment}.png').is_file() or
                    metrics['checkpoint_sha256'] != protocol['checkpoint_hashes'][experiment] or
                    metrics['test_data_hashes'] != test_hashes):
                raise ValueError('Partial final artifact is incomplete or incompatible')
            results[experiment] = metrics
            summary.append({'experiment': experiment, **metrics['mean'], **metrics['gpu_memory']})
            continue
        model = build_model(experiment).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        metrics, samples = evaluate_model(model, loader, device, manifest,
                                         preview_ids(manifest, 'test'), repeats=5)
        metrics.update(experiment=experiment, selected_epoch=checkpoint['completed_epochs'],
                       manifest_sha256=MANIFEST_SHA256, checkpoint_sha256=digest(run_dir / experiment / 'best.pt'),
                       test_data_hashes=test_hashes, versions=checkpoint['versions'], test_evaluated=True)
        save_comparisons(samples, staging / f'{experiment}.png')
        json_save(metrics, completed_file)
        results[experiment] = metrics
        summary.append({'experiment': experiment, **metrics['mean'], **metrics['gpu_memory']})
        del model
    with (staging / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    json_save(results, staging / 'metrics.json')
    json_save({'complete': True, 'experiments': list(EXPERIMENTS)}, staging / 'complete.json')
    staging.replace(destination)
    return summary
