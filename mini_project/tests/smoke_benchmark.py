"""Bounded real-data CPU checks. Never decode/evaluate held-out test images."""
from pathlib import Path
import json
import argparse
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
import train
from benchmark import (EXPERIMENTS, digest, load_manifest, final_benchmark,
                       MANIFEST_SHA256, json_save, verify_selected_checkpoint, prepare_final_staging)
from losses import DenoisingLoss


def main(output):
    torch.set_num_threads(1)
    manifest = load_manifest(PROJECT)
    test_ids = set(manifest['splits']['test'])
    original_open = Image.open
    opened = set()

    def guarded_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            if Path(path).name in test_ids:
                raise AssertionError(f'Test image was opened: {path}')
            opened.add(Path(path).name)
        return original_open(path, *args, **kwargs)

    results = {}
    output = Path(output)
    output.mkdir(exist_ok=False)
    with patch('PIL.Image.open', side_effect=guarded_open):
        grid = torch.arange(300 * 310, dtype=torch.float32).reshape(1, 1, 300, 310)
        a, b, (top, left, height, width) = train.aligned_crop(grid, grid + 1, 256, torch.Generator().manual_seed(42))
        assert torch.equal(a, grid[..., top:top+height, left:left+width])
        assert torch.equal(b, a+1), 'Noisy/clean crops lost alignment'
        loss = DenoisingLoss(0.1)(torch.ones(1, 1, 32, 32), torch.ones(1, 1, 32, 32))
        assert all(value.item() == 0 for value in loss.values())
        x = torch.rand(1, 1, 32, 32, requires_grad=True)
        loss = DenoisingLoss(0.1)(x, torch.zeros_like(x))
        loss['total'].backward()
        assert loss['edge'] > 0 and torch.isfinite(x.grad).all()
        assert torch.allclose(loss['total'], loss['l1'] + 0.1 * loss['edge'])

        for experiment in EXPERIMENTS:
            train.seed_everything(42)
            model = train.build_model(experiment).eval()
            with torch.no_grad():
                for shape in ((420, 540), (258, 540), (259, 541)):
                    image = torch.rand(1, 1, *shape)
                    prediction = model(image)
                    assert prediction.shape == image.shape and torch.isfinite(prediction).all()
            config = train.TrainConfig(experiment)
            summary = train.train_experiment(PROJECT, output / experiment, config, require_cuda=False)
            assert summary['steps'] == 1 and summary['test_evaluated'] is False
            assert len(summary['validation']['rows']) == 2
            for name in ('best.pt', 'last.pt', 'config.json', 'step_log.json', 'validation.json',
                         'metrics.json', 'loss.png', 'comparisons.png'):
                assert (output / experiment / name).stat().st_size > 0
            for name in ('loss.png', 'comparisons.png'):
                with original_open(output / experiment / name) as image:
                    image.verify()
            checkpoint = torch.load(output / experiment / 'last.pt', weights_only=True)
            assert checkpoint['history'][0]['grad_norm'] > 0
            initial = train.build_model(experiment)
            train.seed_everything(42)
            initial = train.build_model(experiment)
            assert any(not torch.equal(value.cpu(), initial.state_dict()[key])
                       for key, value in checkpoint['model_state_dict'].items())
            results[experiment] = summary
        checkpoints = [torch.load(output / name / 'last.pt', weights_only=True) for name in EXPERIMENTS]
        assert len({(c['history'][0]['id'], tuple(c['history'][0]['crop'])) for c in checkpoints}) == 1
        assert checkpoints[2]['initial_state_sha256'] == checkpoints[3]['initial_state_sha256']
        assert checkpoints[2]['config']['edge_weight'] == checkpoints[3]['config']['edge_weight']
        best = torch.load(output / 'cnn_l1/best.pt', weights_only=True)
        verify_selected_checkpoint(checkpoints[0], best)
        try:
            verify_selected_checkpoint(checkpoints[0], {**best, 'completed_epochs': 0})
        except ValueError:
            pass
        else:
            raise AssertionError('Stale best checkpoint accepted')

        # Actual interrupted-epoch recovery, including a missing best.pt. These
        # are four additional optimizer updates total, all bounded smoke mode.
        with tempfile.TemporaryDirectory(prefix='aie464_resume_') as temp:
            temp = Path(temp)
            config = train.TrainConfig('cnn_l1', epochs=2)
            train.train_experiment(PROJECT, temp / 'reference', config, require_cuda=False)
            actual_save = train.checkpoint_save

            class SimulatedDisconnect(Exception):
                pass

            def disconnect(checkpoint, path):
                actual_save(checkpoint, path)
                if Path(path).name == 'last.pt' and checkpoint['completed_epochs'] == 1:
                    raise SimulatedDisconnect()

            with patch('train.checkpoint_save', side_effect=disconnect):
                try:
                    train.train_experiment(PROJECT, temp / 'resumed', config, require_cuda=False)
                except SimulatedDisconnect:
                    pass
                else:
                    raise AssertionError('Disconnect simulation did not run')
            assert not (temp / 'resumed/best.pt').exists()
            train.train_experiment(PROJECT, temp / 'resumed', config, resume=True, require_cuda=False)
            reference = torch.load(temp / 'reference/last.pt', weights_only=True)
            resumed = torch.load(temp / 'resumed/last.pt', weights_only=True)
            for key, value in reference['model_state_dict'].items():
                torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
            for a, b in zip(reference['history'], resumed['history']):
                assert [a[k] for k in ('id', 'crop', 'total', 'l1', 'edge')] == [b[k] for k in ('id', 'crop', 'total', 'l1', 'edge')]
            # Artifact-only final staging checks: no Test Dataset or metrics.
            staging = prepare_final_staging(temp, {'run': 'synthetic-artifact-check'})
            assert prepare_final_staging(temp, {'run': 'synthetic-artifact-check'}) == staging
            try:
                prepare_final_staging(temp, {'run': 'changed'})
            except ValueError:
                pass
            else:
                raise AssertionError('Changed final staging protocol accepted')
            marker_case = temp / 'marker_interruption'
            (marker_case / 'final_test.partial').mkdir(parents=True)
            (marker_case / 'final_test.partial/protocol.json.tmp').write_text('{truncated')
            recovered = prepare_final_staging(marker_case, {'run': 'marker-recovery'})
            assert json.loads((recovered / 'protocol.json').read_text()) == {'run': 'marker-recovery'}
            # Changed config must not silently resume into a different experiment.
            try:
                train.train_experiment(PROJECT, temp / 'resumed', train.TrainConfig('cnn_l1'),
                                       resume=True, require_cuda=False)
            except ValueError:
                pass
            else:
                raise AssertionError('Changed resume config accepted')

            # Disconnect immediately after epoch-0 checkpoint, before any update.
            def first_disconnect(checkpoint, path):
                actual_save(checkpoint, path)
                if checkpoint['completed_epochs'] == 0:
                    raise SimulatedDisconnect()

            config = train.TrainConfig('cnn_l1')
            with patch('train.checkpoint_save', side_effect=first_disconnect):
                try:
                    train.train_experiment(PROJECT, temp / 'first_epoch', config, require_cuda=False)
                except SimulatedDisconnect:
                    pass
                else:
                    raise AssertionError('Initial disconnect simulation did not run')
            train.train_experiment(PROJECT, temp / 'first_epoch', config, resume=True, require_cuda=False)
            initial_resumed = torch.load(temp / 'first_epoch/last.pt', weights_only=True)
            for key, value in checkpoints[0]['model_state_dict'].items():
                torch.testing.assert_close(value, initial_resumed['model_state_dict'][key], rtol=0, atol=0)

        for action in (lambda: final_benchmark(PROJECT, output),
                       lambda: train.train_experiment(PROJECT, output / 'blocked',
                                                       train.TrainConfig('cnn_l1', mode='train', epochs=20),
                                                       require_cuda=False)):
            try:
                action()
            except PermissionError:
                pass
            else:
                raise AssertionError('Approval gate failed')
        try:
            train.train_experiment(PROJECT, output / 'cnn_l1', train.TrainConfig('cnn_l1'), require_cuda=False)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing run was overwritten')
    assert digest(PROJECT / 'outputs/phase1/split_manifest.json') == MANIFEST_SHA256
    assert not opened & test_ids
    evidence = {'passed': True, 'device': 'cpu', 'torch': str(torch.__version__),
                'optimizer_updates': 9, 'test_images_opened': 0, 'manifest_unchanged': True,
                'checks': ['native and odd shapes', 'aligned crop', 'finite forward/backward',
                           'Sobel loss', 'artifacts and checkpoint reload', 'same page/crop sequence',
                           'identical U-Net initialization', 'exact interrupted-resume weights/loss',
                           'first-epoch resume', 'stale best checkpoint guard',
                           'final artifact staging resume/protocol guard (no Test evaluation)',
                           'approval gates', 'overwrite guard', 'resume config guard'],
                'results': results, 'quality_claim': False}
    json_save(evidence, output / 'smoke_checks.json')
    print(json.dumps({k: v for k, v in evidence.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/final_benchmark_smoke')
    main(parser.parse_args().output)
