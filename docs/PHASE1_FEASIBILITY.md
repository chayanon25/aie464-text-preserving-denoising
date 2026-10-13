# Phase 1 — Real-data dataset feasibility (2026-10-12)

> **เอกสารประวัติการวางแผน/พัฒนา:** เนื้อหาและสถานะด้านล่างบันทึกไว้ก่อนชุด Final และอาจต่างจาก implementation จริง ผลล่าสุดใช้ [Final Notebook](../mini_project/FINAL_SUBMISSION/aie464-text-preserving-image-denoising-final.ipynb) และ [Presentation Brief](PRESENTATION_BRIEF.md) เป็นหลัก Notebook รุ่นพัฒนาที่กล่าวถึงเก็บไว้ในเครื่องและไม่รวมใน GitHub

Status: real-data pairing, DataLoader and 10-pair visual review completed.
The data pipeline is feasible. A seed-42 group-aware split now keeps all four
variants of each identical clean document together, with zero group overlap.
The previous provisional filename split is superseded and must not be used.
No models were created or trained. No dependencies were installed.

## Access and downloads

Authenticated browser access was confirmed for the Kaggle account Thames after
user acceptance of the competition rules. Only the two approved training ZIPs
were downloaded through their individual file buttons; neither `test.zip` nor
`sampleSubmission.csv.zip` was downloaded. Both ZIP CRC checks passed.

| Archive | Bytes | SHA256 |
|---|---:|---|
| train.zip | 19,407,030 | 81bab2528971a846be449ad1543b0228c9bb2ed0416f8cf38dd1de4ec3c23720 |
| train_cleaned.zip | 6,605,986 | c8d34fe9e82d012f3c6e474e5296d0c2efc4f7fd11508dba6b5871749c4f7b80 |

Archives and extracted images are in `mini_project/data/raw`. Browser downloads
also remain in the user's Downloads folder. Kaggle lists all four competition
archives at 58.81 MB; the two obtained here total 26,013,016 bytes.

The [official Kaggle data page](https://www.kaggle.com/competitions/denoising-dirty-documents/data)
explicitly describes synthetic noise added to text images to simulate messy
artifacts. This corrects the implication in `docs/DATASET.md` that the degradation
is verified real-world noise; that specification was left unchanged.

## Real-data results

- Exactly 144 noisy PNGs and 144 clean PNGs; exact filenames match with no
  missing pairs. Every pair has equal spatial dimensions.
- 96 pairs are width 540 × height 420; 48 are width 540 × height 258.
  No resizing or augmentation was applied.
- Grayscale tensors are float32 `[1,H,W]`, finite, and in `[0,1]`.
  Global observed ranges are `[0,1]` for both noisy and clean images.
- DataLoader traversal passed for all 144 pairs: 100 train, 20 validation,
  24 test, with batch size 1 and zero workers. The first batch was
  `[1,1,420,540]`; its paired tensors occupied 1,814,400 bytes (~1.73 MiB).
- The original audit used seed-42 whole-page filename splits, before any crops.
  That split had repeated clean content across sets and has now been replaced
  by the group-aware manifest below. Original previews remain historical evidence.
- CPU only; no GPU memory allocated by this pipeline. No Colab/T4 model-memory
  claim is made. DataLoader behavior and dependencies are compatible with Colab.

Reviewed training-page IDs: `153.png`, `2.png`, `20.png`, `23.png`, `47.png`,
`5.png`, `50.png`, `56.png`, `60.png`, `95.png`.

All 10 were inspected in full-page side-by-side views and magnified central
ink overlays. Text content, line positions, and glyph edges match, with no
obvious translation, rotation, or scale mismatch. Noisy pages show background
texture, stains/shadows and crease-like artifacts; clean targets retain the
text at corresponding locations. This is visual alignment evidence for those
10 pairs, not a measured subpixel registration test or visual review of all 144.

## Original finding: repeated clean documents

SHA256 hashes of decoded grayscale pixels plus dimensions identify **36 unique
clean images**, each appearing under four different filenames. **27 groups
span multiple splits** in the provisional 100/20/24 filename manifest.
For example, `128.png`, `125.png`, `122.png`, `131.png` share the exact same clean
image and occur across train, validation and test.

Whole-filename splitting prevents crop leakage but does not prevent source-page
leakage here. The original split must NOT be used for model selection or final
benchmarks. No training occurred, so no model contamination occurred.
The ten selected filenames also include repeated targets; they are ten pairs,
not ten independent source documents.

## Implemented group-aware split

The authoritative split is `mini_project/outputs/phase1/split_manifest.json`.
Each group ID is SHA256 of two big-endian uint32 dimensions (width, height)
followed by decoded grayscale uint8 pixels in row-major order. All four paired
filenames sharing that clean page stay together. Grouping happens on full pages,
before any future patch cropping.

The implementation sorts group hashes, permutes them with NumPy PCG64 seed 42,
and assigns the first 25 groups to train, next 5 to validation, and final 6 to test.
Members and per-split filenames are sorted; the manifest contains the method,
seed, complete group membership, group assignments, pair IDs, counts, and checks.

| Split | Unique clean-document groups | Noisy/Clean pairs |
|---|---:|---:|
| Train | 25 | 100 |
| Validation | 5 | 20 |
| Test | 6 | 24 |
| Total | 36 | 144 |

Verification passed:

- Zero clean-group and filename overlap for train/val, train/test, and val/test.
  Independent hashes of decoded pixels confirmed content separation.
- Every group has exactly four variants, all assigned to one split; all 144
  paired IDs are covered exactly once.
- Reversed input order gives the same manifest. Regeneration with seed 42 is
  byte-identical. Saving rejects an existing manifest with different contents.
- DataLoader traversal of all 100/20/24 pairs passed shape, float32, finite-value,
  and `[0,1]` checks. Partial inputs and incorrect group structures were rejected.
- Code review found no important defects. No models were trained, no data was
  redownloaded, and no dependencies were installed during this split change.

Manifest SHA256:
`5b569d65a4d21aacff84e25986fce522847ce61a4575cd720cd60ca4a4180539`.

The guarantee covers identical decoded clean documents; near-duplicate text or
layout families remain a limitation of exact grouping. Original exploratory
previews were selected before the group-aware split and are retained as evidence
of that earlier review, not as previews selected from the new train assignment.

## Suitability and remaining risks

Provisionally suitable for a small supervised text-denoising experiment:
paired targets are accessible, dimensions align, noise is visible, and the
PyTorch loader works. Exact clean-document leakage is resolved by the grouped
manifest. Only 36 distinct clean images limits independent diversity; synthetic
noise does not establish generalization to real camera/scanner degradation.
Clean image targets do not provide independently transcribed OCR ground truth,
so later CER evaluation needs reviewed reference text and a predeclared protocol.
The public Kaggle test lacks clean targets according to the specification;
use a leakage-safe held-out subset of labeled data for shared model comparisons.

## Files and reproduction

The earlier inspection read `AGENTS.md`, `docs/PROJECT.md`, and `docs/DATASET.md`.
The workspace holds course materials in `W2`–`W7`, planning references in
`mini_project`, and is not a Git repository. No unrelated files were modified.
The minimal implementation remains `mini_project/src/dataset.py` and
`mini_project/src/feasibility.py`. Existing runtime versions: PyTorch 2.8.0,
NumPy 1.26.4, Pillow 10.4.0, Matplotlib 3.9.4.

From the workspace root:

```bash
python3 mini_project/src/feasibility.py \
  --noisy-dir mini_project/data/raw/train \
  --clean-dir mini_project/data/raw/train_cleaned \
  --output mini_project/outputs/phase1 --split-only
```

Artifacts in `mini_project/outputs/phase1`:

- `split_manifest.json`: authoritative group-aware assignments for all later
  models; load its `splits.train`, `splits.val`, and `splits.test` IDs into
  `PairedDocumentDataset`. Use the same held-out test IDs for every model.
- `audit.json`: historical page dimensions/ranges, superseded filename splits,
  batch evidence, preview IDs, and original visual review.
- `pairs_10.png`: historical noisy/clean/ink-overlay full-page visualization.
- `alignment_crops_10.png`: magnified views of those same 10 historical pairs.
- `duplicates.json`: exact duplicate groups and historical cross-split groups.
- `review.json`: historical visual findings, DataLoader traversal, and warning
  for the superseded filename split.

`--split-only` regenerates/verifies the group-aware manifest without altering
the historical audit or previews. Omitting that flag runs the full audit with
the grouped split and generates new train-only previews; the new previews need
their own visual review. Both paths use the same grouping logic and refuse to
overwrite a different existing split manifest. Historical artifacts were not
regenerated during this change.
