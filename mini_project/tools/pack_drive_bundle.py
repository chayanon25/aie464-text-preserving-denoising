"""Pack everything a Colab run needs into aie464_data.zip (upload to MyDrive/AIE464_runs/data/).

Layout inside the zip (unzipped to /content/data on Colab):
  raw/train, raw/train_cleaned                 Kaggle pages
  manifests/split_manifest.json, manifests/dibco_split.json
  external/dibco/dibco{2017,2018,2019}/...     .bmp only (no .dat weights, no archives)
  external/augraphy_shift/...                  held-out shift test set
  checkpoints/<exp>/best.pt                    4 midterm checkpoints
  nafnet/NAFNet-SIDD-width32.pth               only if downloaded
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bundle_entries(root=ROOT):
    data = root / 'data'
    entries = []
    for kind in ('train', 'train_cleaned'):
        entries += [(p, f'raw/{kind}/{p.name}') for p in sorted((data / 'raw' / kind).glob('*.png'))]
    entries.append((root / 'outputs' / 'phase1' / 'split_manifest.json', 'manifests/split_manifest.json'))
    entries.append((root / 'outputs' / 'final' / 'dibco_split.json', 'manifests/dibco_split.json'))
    dibco = data / 'external' / 'dibco'
    entries += [(p, f'external/dibco/{p.relative_to(dibco).as_posix()}') for p in sorted(dibco.rglob('*.bmp'))]
    shift = data / 'external' / 'augraphy_shift'
    entries += [(p, f'external/augraphy_shift/{p.relative_to(shift).as_posix()}')
                for p in sorted(shift.rglob('*')) if p.is_file()]
    ckpts = root / 'backups' / 'aie464_final_artifacts' / 'checkpoints'
    entries += [(p, f'checkpoints/{p.parent.name}/best.pt') for p in sorted(ckpts.glob('*/best.pt'))]
    nafnet = data / 'external' / 'nafnet' / 'NAFNet-SIDD-width32.pth'
    if nafnet.exists():
        entries.append((nafnet, 'nafnet/NAFNet-SIDD-width32.pth'))
    missing = [str(src) for src, _ in entries if not src.exists()]
    if missing:
        raise FileNotFoundError(f'Missing bundle inputs: {missing[:5]}')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'final' / 'aie464_data.zip'))
    args = parser.parse_args()
    entries = bundle_entries()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in entries:
            zf.write(src, arc)
    print(f'Wrote {out} ({out.stat().st_size / 2**20:.1f} MB, {len(entries)} files)')


if __name__ == '__main__':
    main()
