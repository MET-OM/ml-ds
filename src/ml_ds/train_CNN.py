import argparse
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from ml_ds.dataset import ERA5ZarrDataset, build_split_subsets
from ml_ds.models import ConvResNet
from ml_ds.network import LightningModule

BATCH_SIZE = 32
MAX_EPOCHS = 100
NUM_WORKERS = 8
LEARNING_RATE = 1e-3

DATA_ROOT = Path.home() / "ml-ds_data" / "input_data"
INPUT_FILE = DATA_ROOT / "2011.zarr"
STATS_FILE = DATA_ROOT / "era5_normalization_stats.zarr"

CRITERION = nn.L1Loss()


def load_datasets() -> tuple[ERA5ZarrDataset, object, object]:
    full_train_dataset = ERA5ZarrDataset(INPUT_FILE, STATS_FILE)
    val_dataset, test_dataset = build_split_subsets(full_train_dataset)
    return full_train_dataset, val_dataset, test_dataset


def initialize_model(in_channels: int, out_channels: int) -> ConvResNet:
    return ConvResNet(
        in_channels=in_channels,
        out_channels=out_channels,
        n_filters=100,
        n_blocks=2,
        normalization="batch",
        dropout_rate=0.1,
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
        "--enable-validation",
        action="store_true",
        help="Enable validation loop. Default is disabled (train-only).",
    )
    args = parser.parse_args()

    train_data, val_data, test_data = load_datasets()
    print(f"Training data: {len(train_data)} samples.")
    print(f"Validation pipeline ready: {len(val_data)} samples.")
    print(f"Test pipeline ready: {len(test_data)} samples.")

    print(f"Input channels: {len(train_data.input_vars)} -> {train_data.input_vars}")
    print(f"Target channels: {len(train_data.target_vars)} -> {train_data.target_vars}")

    model = initialize_model(
        in_channels=len(train_data.input_vars),
        out_channels=len(train_data.target_vars),
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
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            criterion=CRITERION,
            enable_validation=args.enable_validation,
        )
    else:
        network = LightningModule(
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            lr=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            criterion=CRITERION,
            enable_validation=args.enable_validation,
        )

    # Configure checkpoint callback to save every epoch
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
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
        num_sanity_val_steps=0,
        limit_val_batches=0 if not args.enable_validation else 1.0,
    )
    trainer.fit(network)


if __name__ == "__main__":
    main()
