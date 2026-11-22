"""Phase 1 paired pages: no resizing, augmentation, cropping, or downloads."""

from pathlib import Path
import hashlib
import json
import struct

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def paired_ids(noisy_dir, clean_dir):
    """Require an exact filename match between the two PNG directories."""
    folders = [Path(noisy_dir), Path(clean_dir)]
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing image directory: {folder}. Supply extracted data; no download is performed.")
    names = [{p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.png'} for folder in folders]
    if names[0] != names[1]:
        raise ValueError(f"Unpaired files: missing clean={sorted(names[0] - names[1])}; missing noisy={sorted(names[1] - names[0])}")
    if not names[0]:
        raise ValueError("No paired PNG images found")
    return sorted(names[0])


def create_split_manifest(ids, clean_dir, seed=42):
    """Group identical decoded clean pages before splitting their four variants."""
    ids = list(ids)
    if len(ids) != 144 or len(set(ids)) != 144:
        raise ValueError("Expected exactly 144 unique paired page IDs for the 100/20/24 protocol")
    groups = {}
    for page_id in sorted(ids):
        if Path(page_id).name != page_id:
            raise ValueError("Page IDs must be filenames, not paths")
        with Image.open(Path(clean_dir) / page_id) as image:
            pixels = np.array(image.convert('L'), dtype=np.uint8)
        height, width = pixels.shape
        group_id = hashlib.sha256(struct.pack('>II', width, height) + pixels.tobytes()).hexdigest()
        groups.setdefault(group_id, []).append(page_id)
    if len(groups) != 36 or any(len(members) != 4 for members in groups.values()):
        raise ValueError("Expected exactly 36 clean-document groups with four variants each")
    shuffled = np.random.default_rng(seed).permutation(sorted(groups)).tolist()
    group_splits = {'train': shuffled[:25], 'val': shuffled[25:30], 'test': shuffled[30:]}
    splits = {name: sorted(page_id for group_id in selected for page_id in groups[group_id])
              for name, selected in group_splits.items()}
    names = ('train', 'val', 'test')
    group_overlap = {f'{left}/{right}': len(set(group_splits[left]) & set(group_splits[right]))
                     for i, left in enumerate(names) for right in names[i + 1:]}
    page_overlap = {f'{left}/{right}': len(set(splits[left]) & set(splits[right]))
                    for i, left in enumerate(names) for right in names[i + 1:]}
    if any(group_overlap.values()) or any(page_overlap.values()) or set(sum(splits.values(), [])) != set(ids):
        raise ValueError("Split verification failed: overlap or incomplete coverage")
    return {'schema_version': 1, 'seed': seed,
            'algorithm': 'NumPy default_rng (PCG64), permute sorted clean hashes, 25/5/6 groups',
            'group_hash_method': 'SHA256: big-endian uint32 width,height followed by decoded grayscale uint8 row-major pixels',
            'groups': dict(sorted(groups.items())), 'group_splits': group_splits, 'splits': splits,
            'counts': {name: {'groups': len(group_splits[name]), 'pairs': len(splits[name])} for name in names},
            'checks': {'group_overlap': group_overlap, 'page_overlap': page_overlap,
                       'all_144_pairs_covered': True, 'four_variants_per_group': True}}


def save_split_manifest(manifest, path):
    """Preserve an existing split; never silently replace different assignments."""
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"Existing split manifest differs; refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')


class PairedDocumentDataset(Dataset):
    """Return noisy, clean float32 tensors [1,H,W] in [0,1], plus page ID."""

    def __init__(self, noisy_dir, clean_dir, ids=None):
        self.noisy_dir, self.clean_dir = Path(noisy_dir), Path(clean_dir)
        available = paired_ids(noisy_dir, clean_dir)
        self.ids = available if ids is None else list(ids)
        if not self.ids or len(set(self.ids)) != len(self.ids) or not set(self.ids).issubset(available):
            raise ValueError("Page IDs must be nonempty, unique, and present in both directories")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        page_id = self.ids[index]
        tensors = []
        for folder in (self.noisy_dir, self.clean_dir):
            with Image.open(folder / page_id) as image:
                pixels = np.array(image.convert('L'), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(pixels).unsqueeze(0))
        noisy, clean = tensors
        if noisy.shape != clean.shape:
            raise ValueError(f"Dimension mismatch for {page_id}: {tuple(noisy.shape)} vs {tuple(clean.shape)}")
        return {'noisy': noisy, 'clean': clean, 'id': page_id}


def make_loader(dataset, shuffle=False, seed=42):
    """Batch 1 preserves variable page dimensions; CPU only, notebook safe."""
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0,
                      pin_memory=False, generator=torch.Generator().manual_seed(seed))


def _load_gray_array(path):
    with Image.open(path) as image:
        return np.array(image.convert('L'), dtype=np.float32) / 255.0


def load_kaggle_split(root, manifest, split):
    """Load one split of the frozen Kaggle manifest as numpy samples.

    Each sample: dict(id, noisy[H,W] float32, clean[H,W] float32, group).
    """
    root, manifest = Path(root), json.loads(Path(manifest).read_text())
    noisy_dir, clean_dir = root / 'train', root / 'train_cleaned'
    available = set(paired_ids(noisy_dir, clean_dir))
    group_of = {page: group for group, pages in manifest['groups'].items() for page in pages}
    ids = manifest['splits'][split]
    missing = sorted(set(ids) - available)
    if missing:
        raise ValueError(f'Manifest pages missing from {root}: {missing[:5]}')
    samples = []
    for page_id in ids:
        noisy, clean = _load_gray_array(noisy_dir / page_id), _load_gray_array(clean_dir / page_id)
        if noisy.shape != clean.shape:
            raise ValueError(f'Dimension mismatch for {page_id}')
        samples.append({'id': page_id, 'noisy': noisy, 'clean': clean, 'group': group_of[page_id]})
    return samples


class RandomPatchDataset(Dataset):
    """Random square patches from numpy samples; deterministic per (seed, epoch, index).

    Pages smaller than the patch are padded with white (1.0) on both images.
    `augment(noisy, clean, rng) -> (noisy, clean)` runs after cropping.
    """

    def __init__(self, samples, patch_size=256, samples_per_epoch=1000, augment=None, seed=42):
        if not samples:
            raise ValueError('RandomPatchDataset needs at least one sample')
        self.samples, self.patch_size = list(samples), int(patch_size)
        self.samples_per_epoch, self.augment, self.seed = int(samples_per_epoch), augment, int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, self.epoch, int(index)])
        sample = self.samples[int(rng.integers(len(self.samples)))]
        p = self.patch_size
        noisy, clean = sample['noisy'], sample['clean']
        height, width = noisy.shape
        pad = ((0, max(0, p - height)), (0, max(0, p - width)))
        if pad[0][1] or pad[1][1]:
            noisy = np.pad(noisy, pad, constant_values=1.0)
            clean = np.pad(clean, pad, constant_values=1.0)
            height, width = noisy.shape
        top, left = int(rng.integers(height - p + 1)), int(rng.integers(width - p + 1))
        noisy = noisy[top:top + p, left:left + p]
        clean = clean[top:top + p, left:left + p]
        if self.augment is not None:
            noisy, clean = self.augment(noisy, clean, rng)
        return (torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(clean, dtype=np.float32))[None])
