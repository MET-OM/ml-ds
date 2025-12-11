import argparse
from pathlib import Path

from pytorch_lightning import Trainer

from ml_ds.dataset import ERA5dataset
from ml_ds.models import ConvResNet
from ml_ds.network import LightningModule

BATCH_SIZE = 20
MAX_EPOCHS = 5

STATIC_VARS = ["land_mask"]
INPUT_VARS = ["u10", "v10"]
TARGET_VARS = ["u10", "v10"]

# Selecting data. Here, we just use one file (year) each for train, val and test.
DIR_DATA = str(Path.home() / "ml-ds_data")
INPUT_FILES = [DIR_DATA + f"/ERA5_{y}_reinterp.nc" for y in [1940, 1941, 1942]]
TARGET_FILES = [DIR_DATA + f"/ERA5_{y}.nc" for y in [1940, 1941, 1942]]
STATIC_DATA = DIR_DATA + "/GEBCO_gridded.nc"

TRAIN_INPUT = [INPUT_FILES[0]]
VAL_INPUT = [INPUT_FILES[1]]
TEST_INPUT = [INPUT_FILES[2]]

TRAIN_TARGET = [TARGET_FILES[0]]
VAL_TARGET = [TARGET_FILES[1]]
TEST_TARGET = [TARGET_FILES[2]]


def load_datasets():
    # Create datasets
    train_data = ERA5dataset(
        TRAIN_INPUT, INPUT_VARS, TRAIN_TARGET, TARGET_VARS, STATIC_DATA, STATIC_VARS
    )
    val_data = ERA5dataset(
        VAL_INPUT, INPUT_VARS, VAL_TARGET, TARGET_VARS, STATIC_DATA, STATIC_VARS, train_data.mean_sd
    )
    test_data = ERA5dataset(
        TEST_INPUT,
        INPUT_VARS,
        TEST_TARGET,
        TARGET_VARS,
        STATIC_DATA,
        STATIC_VARS,
        train_data.mean_sd,
    )
    return train_data, val_data, test_data


def initialize_model():
    # Initialize model
    return ConvResNet(
        in_channels=len(INPUT_VARS) + len(STATIC_VARS),
        out_channels=len(TARGET_VARS),
        n_filters=40,
        n_blocks=8,
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
    args = parser.parse_args()

    train_data, val_data, test_data = load_datasets()
    print(f"Training data: {len(train_data)} samples.")
    print(f"Validation data: {len(val_data)} samples.")
    print(f"Test data: {len(test_data)} samples.")

    model = initialize_model()
    print(model)

    if args.checkpoint:
        network = LightningModule.load_from_checkpoint(
            args.checkpoint,
            model=model,
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            batch_size=BATCH_SIZE, 
            num_workers=8,
        )
    else:
        network = LightningModule(
            model, train_data, val_data, test_data, batch_size=BATCH_SIZE, num_workers=8
        )

    trainer = Trainer(profiler="simple", max_epochs=MAX_EPOCHS)
    trainer.fit(network)

if __name__ == "__main__":
    main()
