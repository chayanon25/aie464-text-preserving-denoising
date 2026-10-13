# Phase 2A — Minimal CNN sanity notebook

The notebook is implemented and locally smoke-tested. It has no saved outputs
or benchmark claims. A Colab T4 run is still pending; no full training was run.

## Implementation plan

1. Reuse `dataset.py` and the frozen group-aware `split_manifest.json`; add a
   five-convolution CNN and dependency-free PyTorch PSNR/SSIM helpers.
2. Train at most 20 steps with L1/Adam on aligned 256×256 training crops, then
   evaluate full validation pages and save comparisons, logs and final weights.
3. Validate notebook syntax, metric calculations, split isolation, one-step
   CPU training, checkpoint reload and same-environment reproducibility.

## Files

- `mini_project/notebooks/02a_cnn_sanity_colab.ipynb`: Colab setup, brief training,
  validation, plots, logs and checkpoint saving.
- `mini_project/src/models/shallow_cnn.py`: five 3×3 convolutions, width 16,
  ReLU hidden activations, sigmoid output; 7,265 parameters. Input/output spatial
  dimensions are equal. No U-Net or Attention.
- `mini_project/src/evaluate.py`: per-page PSNR and SSIM for grayscale `[0,1]`.
  SSIM uses an 11×11 Gaussian kernel, sigma 1.5, valid windows, population
  covariance, K1=.01 and K2=.03, without downsampling.

Existing data files, split manifest, Dataset/DataLoader and Phase 1 artifacts
were not changed. No downloads or dependency installations occurred.

## Run in Colab

1. Upload the notebook to Colab and choose **Runtime → Change runtime type →
   T4 GPU**.
2. Copy the existing `mini_project/src` directory, extracted
   `data/raw/train` and `data/raw/train_cleaned` folders, and
   `outputs/phase1/split_manifest.json` to your Google Drive. Preserve those
   relative paths. The existing dataset contains 144 noisy/clean pairs.
3. Run the Drive-mount cell, authorize your own account, then edit `PROJECT_DIR`
   to the copied `mini_project` folder. The default path is
   `/content/drive/MyDrive/AIE464/mini_project`.
4. Run the remaining cells in order in a fresh runtime. Keep the defaults:
   seed 42, width 16, batch size 1, patch size 256, learning rate .001, 20 steps.
   Missing data, dependencies or a CUDA GPU causes an explicit stop. No install
   or download commands are included.

The notebook verifies the frozen manifest checksum and group membership.
Only training and validation images are decoded; test IDs are read as metadata
to verify separation. There is no test Dataset, test metric or test preview.
The final model is saved regardless of validation scores, without a tuning loop.

Each training pair uses the same randomly selected crop coordinates for input
and target, after the group split. All 20 validation pages remain at their native
dimensions. Noisy-input scores and denoised scores are averaged per page over
those same 20 pairs. Three comparison rows use distinct validation document
groups. A 20-step sanity run can score worse than the noisy input and does not
establish convergence or denoising quality.

## Outputs and reproducibility

Outputs go to `mini_project/outputs/phase2a_sanity` on Drive:

- `checkpoint.pt`: final weights, optimizer/scaler state, seeds/RNG states,
  actual page/crop sequence, configuration, versions, module hashes, train/val
  file hashes, manifest, loss history and validation results.
- `step_log.json`: training losses, sampled IDs/crops, elapsed seconds and peak
  allocated/reserved GPU memory.
- `metrics.json`: per-page validation scores and means, train/validation runtime,
  runtime versions and peak GPU memory.
- `loss.png` and `comparisons.png`: training loss and Noisy/Denoised/Clean rows.

The notebook uses `torch.amp.autocast` and `torch.amp.GradScaler` on CUDA;
metrics run in float32 outside autocast. It records peak PyTorch allocated and
reserved memory and stops if either exceeds 6 GiB. The small CNN, batch size 1
and training crops target the project memory budget, but actual T4 consumption
has not been measured. Memory/runtime summaries cover setup, training and
validation through the summary cell, not subsequent saving, plots or reload.
PyTorch memory counters exclude other GPU processes and some driver overhead.

Restarting and running all cells with the same manifest, source, data, settings
and hardware/software environment reproduces the run. Python, NumPy and PyTorch
seeds are fixed; deterministic algorithms are enabled. Different platforms or
versions are not guaranteed bit-identical. The checkpoint supports inference
reload and records reproduction inputs; automatic mid-epoch resume is outside
this minimal notebook. A final cell verifies identical predictions after loading
saved weights with `weights_only=True`.

Existing checkpoints are not overwritten: use a new `OUTPUT_DIR` and restart
the runtime for another sanity run. The notebook enforces a maximum of 20 steps;
full training requires a separate authorized phase.

## Verification evidence

- Notebook schema and every Python cell parsed successfully; all delivered
  execution counts and outputs are empty.
- The CNN preserved both native page sizes, returned finite bounded outputs,
  and passed PSNR identity/known-error tests.
- SSIM passed identity and symmetry checks and agreed within 1e-5 with an
  independent NumPy float64 Gaussian-window calculation. Invalid metric inputs
  were rejected.
- The actual notebook cells ran twice with **one CPU optimizer step** per run
  and all 20 full-page validation pairs. Both runs saved all five artifacts,
  loaded their checkpoints, and reproduced the same validation prediction.
- Repeated CPU runs had exactly equal model weights, selected page, crop
  coordinates, L1 loss (0.4623348713), and per-page validation results.
- Trained IDs belonged only to the training split; validation IDs matched only
  its 20 reserved IDs. Manifest bytes/checksum remained unchanged. Temporary
  CPU checkpoints and plots were deleted after verification.
- Checkpoint overwrite protection passed. Code review found no blocking defects.

Local checks used PyTorch 2.8.0, NumPy 1.26.4, Pillow 10.4.0 and Matplotlib 3.9.4.
They validate the CPU pipeline, not CUDA behavior or T4 memory. No Colab results
have been generated or reported yet.

References: [PyTorch AMP](https://docs.pytorch.org/docs/2.8/amp.html),
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html),
[original SSIM paper](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html).
