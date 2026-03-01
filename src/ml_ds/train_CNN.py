import argparse
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import Dataset

from ml_ds.dataset import ERA5ZarrDataset
from ml_ds.models import ConvResNet
from ml_ds.network import LightningModule

MAX_EPOCHS = 3
NUM_WORKERS = 8
LEARNING_RATE = 1e-3
N_FILTERS = 100
N_BLOCKS = 2
DROPOUT_RATE = 0.1
NORMALIZATION: str | None = "batch"
ACTIVATION = "relu"

DATA_ROOT = Path.home() / "ml-ds_data" / "input_data"
DEFAULT_INPUT_FILE = Path("data.zarr")
DEFAULT_STATS_FILE = Path("era5_normalization_stats.zarr")
DEFAULT_RESULTS_DIR = Path.home() / "ml-ds_results"

CRITERION = nn.L1Loss()


def _resolve_data_path(path: Path, data_root: Path) -> Path:
    return path if path.is_absolute() else data_root / path


def load_datasets(
    input_file: Path,
    stats_file: Path,
    val_input_file: Path | None,
    test_input_file: Path | None,
) -> tuple[Dataset, Dataset | None, Dataset | None]:
    train_dataset = ERA5ZarrDataset(input_file, stats_file)
    val_dataset = (
        ERA5ZarrDataset(val_input_file, stats_file) if val_input_file is not None else None
    )
    test_dataset = (
        ERA5ZarrDataset(test_input_file, stats_file) if test_input_file is not None else None
    )
    return train_dataset, val_dataset, test_dataset


def initialize_model(
    in_channels: int,
    out_channels: int,
    n_filters: int,
    n_blocks: int,
    normalization: str | None,
    activation: str,
    dropout_rate: float,
) -> ConvResNet:
    return ConvResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters=n_filters,
        n_blocks=n_blocks,
        normalization=normalization,
        activation=activation,
        dropout_rate=dropout_rate,
    )


def main():
    parser = argparse.ArgumentParser(description="Train CNN model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root directory for dataset files.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input Zarr path (absolute or relative to --data-root).",
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=DEFAULT_STATS_FILE,
        help="Normalization stats Zarr path (absolute or relative to --data-root).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for checkpoints and training artifacts.",
    )
    parser.add_argument(
        "--val-input-file",
        type=Path,
        default=None,
        help="Optional validation Zarr path (absolute or relative to --data-root).",
    )
    parser.add_argument(
        "--test-input-file",
        type=Path,
        default=None,
        help="Optional test Zarr path (absolute or relative to --data-root).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        seed_everything(args.seed, workers=True)

    data_root = args.data_root.expanduser()
    input_file = _resolve_data_path(args.input_file.expanduser(), data_root)
    stats_file = _resolve_data_path(args.stats_file.expanduser(), data_root)
    val_input_file = (
        _resolve_data_path(args.val_input_file.expanduser(), data_root)
        if args.val_input_file is not None
        else None
    )
    test_input_file = (
        _resolve_data_path(args.test_input_file.expanduser(), data_root)
        if args.test_input_file is not None
        else None
    )

    for required_path in (input_file, stats_file):
        if not required_path.exists():
            raise FileNotFoundError(f"Path does not exist: {required_path}")
    for optional_path in (val_input_file, test_input_file):
        if optional_path is not None and not optional_path.exists():
            raise FileNotFoundError(f"Path does not exist: {optional_path}")

    results_dir = args.results_dir.expanduser()
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_datasets(
        input_file=input_file,
        stats_file=stats_file,
        val_input_file=val_input_file,
        test_input_file=test_input_file,
    )
    enable_validation = val_data is not None

    print(f"Data root: {data_root}")
    print(f"Input file: {input_file}")
    print(f"Stats file: {stats_file}")
    print(f"Validation input file: {val_input_file}")
    print(f"Test input file: {test_input_file}")
    print(f"Results dir: {results_dir}")
    print(f"Seed: {args.seed}")
    print(f"Training data: {len(train_data)} samples.")
    print(f"Validation pipeline ready: {0 if val_data is None else len(val_data)} samples.")
    print(f"Test pipeline ready: {0 if test_data is None else len(test_data)} samples.")

    train_dataset_for_vars = train_data.dataset if hasattr(train_data, "dataset") else train_data
    time_chunk_size = getattr(train_dataset_for_vars, "time_chunk_size", None)
    if time_chunk_size is None or int(time_chunk_size) <= 0:
        raise ValueError(
            "Could not infer a valid time chunk size from the input dataset. "
            "Batch size is required to come from input Zarr chunking."
        )
    batch_size = int(time_chunk_size)
    print(
        "Input channels: "
        f"{len(train_dataset_for_vars.input_vars)} -> {train_dataset_for_vars.input_vars}"
    )
    print(
        "Target channels: "
        f"{len(train_dataset_for_vars.target_vars)} -> {train_dataset_for_vars.target_vars}"
    )
    print(f"Batch size (from time chunking): {batch_size}")

    model = initialize_model(
        in_channels=len(train_dataset_for_vars.input_vars),
        out_channels=len(train_dataset_for_vars.target_vars),
        n_filters=N_FILTERS,
        n_blocks=N_BLOCKS,
        normalization=NORMALIZATION,
        activation=ACTIVATION,
        dropout_rate=DROPOUT_RATE,
    )
    print(model)

    if args.checkpoint:
        network = LightningModule.load_from_checkpoint(
            args.checkpoint,
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            lr=LEARNING_RATE,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            criterion=CRITERION,
            enable_validation=enable_validation,
        )
    else:
        network = LightningModule(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            lr=LEARNING_RATE,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            criterion=CRITERION,
            enable_validation=enable_validation,
        )

    # Configure checkpoint callback to save every epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename="epoch={epoch:02d}-step={step}",
        every_n_epochs=1,
        save_top_k=-1,  # Save all checkpoints
        save_last=True,
    )

    trainer = Trainer(
        profiler="simple",
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        default_root_dir=results_dir,
        num_sanity_val_steps=0,
        limit_val_batches=0 if not enable_validation else 1.0,
    )
    trainer.fit(network)


if __name__ == "__main__":
    main()
