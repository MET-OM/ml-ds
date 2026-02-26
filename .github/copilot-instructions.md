# Copilot Instructions for `ml-ds`

## Project purpose and architecture
- This repo trains CNN-based statistical downscaling models with PyTorch Lightning.
- Core runtime flow is: `train_CNN.py` (orchestration) -> `dataset.py` (Zarr I/O + normalization) -> `models.py` (ConvResNet) -> `network.py` (Lightning training loop).
- Keep responsibilities separated:
  - `dataset.py`: data discovery, normalization-stat parsing, sample tensors.
  - `models.py`: pure network layers/blocks only.
  - `network.py`: optimization, logging, dataloaders, scheduler wiring.
  - `train_CNN.py`: file paths, CLI flags, Trainer/callback assembly.

## Data contracts (critical)
- Training data is expected at `~/ml-ds_data/input_data/2011.zarr`.
- Normalization stats are expected at `~/ml-ds_data/input_data/era5_normalization_stats.zarr`.
- Variable conventions in input Zarr:
  - Dynamic predictors: all 3D variables prefixed with `x_`.
  - Static predictors: 2D `x_lsm` and `x_orog`.
  - Targets: exactly 3 variables prefixed with `y_`.
- Stats conventions are flexible in `ZarrNormalizationStats` (stat-dim, `{var}_mean/{var}_std`, or attrs); preserve this compatibility when editing.

## Training behavior conventions
- Current default is **train-only**: validation/test datasets are built but not used unless `--enable-validation` is passed.
- `LightningModule.configure_optimizers()` monitors `train_loss` in train-only mode and `val_loss` when validation is enabled.
- Checkpoints are written each epoch under `checkpoints/` from `ModelCheckpoint` in `train_CNN.py`.

## Developer workflows
- Environment setup:
  - `conda env create -f environment.yml`
  - `conda activate ml-ds`
- Install/editable package is provided via `environment.yml` (`-e .` in pip section).
- Main training entrypoint:
  - `mlds-train`
  - or `python -m ml_ds.train_CNN`
- Resume training:
  - `python -m ml_ds.train_CNN --checkpoint checkpoints/<file>.ckpt`
- Lightweight sanity check used in this repo: `python -m compileall src/ml_ds`

## Project-specific coding patterns
- Prefer explicit failure for unsupported options (see `models.py` factory helpers and `NotImplementedError` for attention).
- Keep variable discovery deterministic (`sorted(...)` for dynamic inputs/targets) to avoid channel-order drift.
- When changing dataset outputs, keep channel counts consistent with model construction in `train_CNN.initialize_model(...)`.
- Reuse existing module boundaries instead of adding cross-file shortcuts.

## Related directories
- `scripts/`: offline data prep/conversion utilities (download/regrid/merge/prep), not the training runtime path.
- `notebooks/`: exploratory analysis and diagnostics.