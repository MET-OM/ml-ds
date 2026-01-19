import os
import sys
import torch
import subprocess
import numpy as np

from torch import nn
from itertools import product
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary

from ml_ds.dataset import ERA5dataset
from ml_ds.models import ConvResNet
from ml_ds.multiscale_CNN import MultiScaleConvResNet
from ml_ds.network import LightningModule

print(subprocess.check_output("nvidia-smi").decode("utf-8"))

data_years = np.arange(1940,2000)
INPUT_FILES  = [f"/lustre/storeB/project/fou/om/EuInterchange/ERA5/south_europe/reinterp/ERA5_{y}_reinterp.nc" for y in data_years]
TARGET_FILES = [f"/lustre/storeB/project/fou/om/EuInterchange/ERA5/south_europe/original/ERA5_{y}.nc" for y in data_years]
STATIC_DATA = "/home/erfur3093/EUINTERCHANGE/GEBCO_south_europe.nc"

TRAIN_INPUT = INPUT_FILES[:3]
VAL_INPUT = [INPUT_FILES[3]]
TEST_INPUT = [INPUT_FILES[4]]

TRAIN_TARGET = TARGET_FILES[:3]
VAL_TARGET = [TARGET_FILES[3]]
TEST_TARGET = [TARGET_FILES[4]]

def create_combos():
    # Generates all combinations of the defined lists of hyperparameters.
    hparam_options = {
        "model_type": ["ConvResNet"],
        "num_channels": [64],
        "num_blocks": [2],
        "dropout": [0.1],
        "batch_norm": [True],
        "lr": [1e-3],
        "input_vars": [["u10", "v10"]],
        "target_vars": [["u10", "v10"]],
        "static_vars": [
            # ["land_mask","elevation"],
            ["lat_field","lon_field"],
            # ["lat_field","lon_field","elevation","interp_distance","land_mask"]
        ],
    }

    # Extract hyperparameter names and list of values
    keys = list(hparam_options.keys())
    values_lists = [hparam_options[k] for k in keys]

    # Cartesian product of all hyperparameter values
    combos = []
    for combo_values in product(*values_lists):
        combo = dict(zip(keys, combo_values))
        combos.append(combo)

    return combos

BATCH_SIZE = 64
MAX_EPOCHS = 100
CRITERION = nn.L1Loss()

# Select month based on task ID.
task_id = int(os.environ["SGE_TASK_ID"])-1
combo = create_combos()[task_id]

# Model hyperparameters
CHANNELS   = combo["num_channels"]
BLOCKS     = combo["num_blocks"]
DROPOUT    = combo["dropout"]
BATCH_NORM = combo["batch_norm"]
MODEL_TYPE = combo["model_type"]
INIT_LR    = combo["lr"]

# Dataset / input-output
INPUT_VARS  = combo["input_vars"]
TARGET_VARS = combo["target_vars"]
STATIC_VARS = combo["static_vars"]

print(combo,flush=True)

# Create datasets
train_data  = ERA5dataset(TRAIN_INPUT, INPUT_VARS, TRAIN_TARGET, TARGET_VARS, STATIC_DATA, STATIC_VARS)
val_data    = ERA5dataset(VAL_INPUT,   INPUT_VARS, VAL_TARGET,   TARGET_VARS, STATIC_DATA, STATIC_VARS, train_data.mean_sd)
test_data   = ERA5dataset(TEST_INPUT,  INPUT_VARS, TEST_TARGET,  TARGET_VARS, STATIC_DATA, STATIC_VARS, train_data.mean_sd)

print(f"Training data: {len(train_data)} samples.")
print(f"Validation data: {len(val_data)} samples.")
print(f"Test data: {len(test_data)} samples.")

if MODEL_TYPE == "ConvResNet":
    model = ConvResNet(
        in_channels=len(INPUT_VARS) + len(STATIC_VARS),
        out_channels=len(TARGET_VARS),
        n_filters=CHANNELS,
        n_blocks=BLOCKS,
        normalization="batch" if BATCH_NORM else None,
        dropout_rate=DROPOUT,
    )
elif MODEL_TYPE == "MultiScaleConvResNet":
    model = MultiScaleConvResNet(
        in_channels=len(INPUT_VARS)+len(STATIC_VARS),
        out_channels=len(TARGET_VARS),
        base_channels=CHANNELS,
        num_resblocks=BLOCKS,
        use_batchnorm=BATCH_NORM,
        dropout=DROPOUT
    )
else:
    raise ValueError(MODEL_TYPE)

network = LightningModule(
    model=model,
    train_dataset=train_data,
    val_dataset=val_data,
    test_dataset=test_data,
    batch_size=BATCH_SIZE, 
    num_workers=8,
    lr=INIT_LR,
    criterion=CRITERION,
)

print(network,flush=True)

logger = CSVLogger(save_dir="/home/erfur3093/EUINTERCHANGE/logs",name="4_static_variables")

hyperparams = {
    "model_type": MODEL_TYPE,
    "num_channels": CHANNELS,
    "num_blocks": BLOCKS,
    "dropout": DROPOUT,
    "batch_norm":BATCH_NORM,
    "input_vars":INPUT_VARS,
    "target_vars":TARGET_VARS,
    "static_vars":STATIC_VARS,

    "model_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
    "criterion":network.criterion.__class__.__name__,
    "batch_size": BATCH_SIZE,
    "lr": INIT_LR,
    "in_channels": len(INPUT_VARS) + len(STATIC_VARS),
    "out_channels": len(TARGET_VARS),
    "max_epochs":MAX_EPOCHS,
    "train_input_files": TRAIN_INPUT,
    "val_input_files": VAL_INPUT,
    "test_input_files": TEST_INPUT,
    "train_target_files": TRAIN_TARGET,
    "val_target_files": VAL_TARGET,
    "test_target_files": TEST_TARGET,
}

logger.log_hyperparams(hyperparams)

trainer = Trainer(
    profiler="simple", 
    logger=logger,
    log_every_n_steps=1,
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    callbacks=[TQDMProgressBar(refresh_rate=0)])

print("TRAINING START",flush=True)

trainer.fit(network)

print("TRAINING COMPLETE",flush=True)

trainer.test(network)

print("TESTING COMPLETE",flush=True)