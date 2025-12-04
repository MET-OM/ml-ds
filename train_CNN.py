import xarray as xr
from pytorch_lightning import Trainer

from dataset import ERA5dataset
from network import LightningModule
from models import ConvResNet

batch_size = 16
max_epochs = 5

static_vars = ["land_mask"]
input_vars = ["u10","v10"]
target_vars = ["u10","v10"]

# Selecting data. Here, we just use one file (year) each for train, val and test.
dir_input = "data/"
dir_target = "data/"
input_files = [dir_input+f"ERA5_{y}.nc" for y in [1940,1941,1942]]
target_files = [dir_target+f"ERA5_{y}_reinterp.nc" for y in [1940,1941,1942]]
static_data = "data/GEBCO_gridded.nc"

train_input = [input_files[0]]
val_input   = [input_files[1]]
test_input  = [input_files[2]]

train_target = [target_files[0]]
val_target   = [target_files[1]]
test_target  = [target_files[2]]

# Create datasets
train_data = ERA5dataset(train_input, input_vars, 
                         train_target,target_vars, 
                         static_data, static_vars)
val_data   = ERA5dataset(val_input,   input_vars, 
                         val_target,  target_vars, 
                         static_data, static_vars,
                         train_data.mean_sd)
test_data  = ERA5dataset(test_input,  input_vars, 
                         test_target, target_vars, 
                         static_data, static_vars,
                         train_data.mean_sd)

print(f"Training data: {len(train_data)} samples.")
print(f"Validation data: {len(val_data)} samples.")
print(f"Test data: {len(test_data)} samples.")

# Initialize model
model = ConvResNet(in_channels=len(input_vars) + len(static_vars),
                   out_channels=len(target_vars),
                   n_filters=8,
                   n_blocks=2,
                   normalization="batch",
                   dropout_rate=0.1)
print(model)

# Initialize pytorch-lightning module and trainer.
network = LightningModule(model,train_data,val_data,test_data,batch_size=batch_size,num_workers=4)
trainer = Trainer(profiler="simple",max_epochs=max_epochs)
trainer.fit(network)

# Run test prediction (just one batch)
test_loader = network.test_dataloader()
x,y = next(iter(test_loader))
yh = network.forward(x)

# Put test result back into xarray with coordinates.
era5_coords = xr.open_dataset(input_files[0])
era5_coords = {
    "valid_time":era5_coords["valid_time"].values[:batch_size],
    "latitude":era5_coords["latitude"].values,
    "longitude":era5_coords["longitude"].values
    }
def to_xarray(tensor,vars):
    tensor = tensor.detach().numpy()
    return xr.Dataset(
        {var:xr.DataArray(tensor[:,var_num,:,:],coords=era5_coords)
         for var_num,var in enumerate(vars)}
    )

ds_x = to_xarray(x,input_vars + static_vars)
ds_y = to_xarray(y,target_vars)
ds_yh = to_xarray(yh,target_vars)

ds_x.to_netcdf("test_input.nc")
ds_y.to_netcdf("test_target.nc")
ds_yh.to_netcdf("test_prediction.nc")