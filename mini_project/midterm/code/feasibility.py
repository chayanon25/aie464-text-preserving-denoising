"""Audit locally supplied Dirty Documents pairs and render ten train pages."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import PairedDocumentDataset, make_loader, create_split_manifest, save_split_manifest


def run(noisy_dir, clean_dir, output, split_only=False):
    torch.manual_seed(42)
    np.random.seed(42)
    pages = PairedDocumentDataset(noisy_dir, clean_dir)
    manifest = create_split_manifest(pages.ids, clean_dir)
    output = Path(output)
    save_split_manifest(manifest, output / 'split_manifest.json')
    splits = manifest['splits']
    if split_only:
        print(f"Saved group-aware split_manifest.json to {output}: {manifest['counts']}; zero group/page overlap")
        return
    records = []
    for pair in pages:
        noisy, clean = pair['noisy'], pair['clean']
        for tensor in (noisy, clean):
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
                raise ValueError(f"Invalid pixel values: {pair['id']}")
        records.append({'id': pair['id'], 'shape_chw': list(noisy.shape),
                        'noisy_range': [noisy.min().item(), noisy.max().item()],
                        'clean_range': [clean.min().item(), clean.max().item()]})
    train = PairedDocumentDataset(noisy_dir, clean_dir, splits['train'])
    batch = next(iter(make_loader(train)))
    selected = sorted(np.random.default_rng(42).choice(train.ids, size=10, replace=False).tolist())
    preview = PairedDocumentDataset(noisy_dir, clean_dir, selected)
    fig, axes = plt.subplots(10, 3, figsize=(12, 30), constrained_layout=True)
    for row, pair in enumerate(preview):
        noisy, clean = pair['noisy'][0].numpy(), pair['clean'][0].numpy()
        # Registered text appears white; red/cyan fringes expose displacement.
        overlay = np.stack([1 - noisy, 1 - clean, 1 - clean], axis=-1)
        for col, (pixels, title) in enumerate(((noisy, 'Noisy'), (clean, 'Clean'), (overlay, 'Ink overlay: red=noisy, cyan=clean'))):
            axes[row, col].imshow(pixels, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axes[row, col].set_title(f"{pair['id']} | {title}", fontsize=9)
            axes[row, col].axis('off')
    fig.suptitle('10 training pages — visual alignment review required', fontsize=14)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'pairs_10.png', dpi=140)
    plt.close(fig)
    report = {'seed': 42, 'paired_pages': len(pages), 'splits': splits,
              'split_manifest': 'split_manifest.json', 'split_checks': manifest['checks'],
              'pages': records, 'preview_ids': selected,
              'batch_shape': list(batch['noisy'].shape),
              'batch_pair_bytes': sum(batch[key].numel() * batch[key].element_size() for key in ('noisy', 'clean')),
              'structural_checks': 'passed: filenames, dimensions, dtype, range, finite values',
              'visual_alignment': 'pending human/agent review of pairs_10.png; equal dimensions do not prove registration',
              'device': 'cpu; no model or training'}
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"Audited {len(pages)} pairs; saved audit.json and pairs_10.png to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--noisy-dir', type=Path, required=True)
    parser.add_argument('--clean-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('mini_project/outputs/phase1'))
    parser.add_argument('--split-only', action='store_true', help='Save grouped splits without regenerating the historical audit or previews')
    args = parser.parse_args()
    try:
        run(args.noisy_dir, args.clean_dir, args.output, split_only=args.split_only)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.exit(1, f"Feasibility check stopped: {error}\n")
