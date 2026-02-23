import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# --- Folders ---
annual_folder = reinterp_folder = str(Path.home() / "ml-ds_data") + "/"

# --- Year range ---
years = np.arange(1940, 1951)

# --- EC-Earth3 grid specification ---
ecearth_file = os.path.join(os.path.dirname(__file__), "EC-Earth3.grid.nc")
ds_ec = xr.open_dataset(ecearth_file)
coarse_lon = ds_ec["lon"].values
coarse_lon = ((coarse_lon + 180) % 360) - 180
coarse_lat = ds_ec["lat"].values

encoding = {
    "t2m": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 273.0,
        "zlib": True,
        "shuffle": True,
        "complevel": 1,
        "chunksizes": (1, 161, 181),
        "_FillValue": np.int16(-32768),
    },
    "d2m": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 273.0,
        "zlib": True,
        "shuffle": True,
        "complevel": 1,
        "chunksizes": (1, 161, 181),
        "_FillValue": np.int16(-32768),
    },
    "u10": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 0.0,
        "zlib": True,
        "shuffle": False,
        "complevel": 1,
        "chunksizes": (1, 161, 181),
        "_FillValue": np.int16(-32768),
    },
    "v10": {
        "dtype": "int16",
        "scale_factor": 0.01,
        "add_offset": 0.0,
        "zlib": True,
        "shuffle": False,
        "complevel": 1,
        "chunksizes": (1, 161, 181),
        "_FillValue": np.int16(-32768),
    },
}


# --- Processing function ---
def process_year(year):
    path_in = f"{annual_folder}ERA5_{year}.nc"
    path_out = f"{reinterp_folder}ERA5_{year}_reinterp.nc"

    if not os.path.exists(path_in):
        print(f"Skipped {year}: input missing")
    if os.path.exists(path_out):
        try:
            xr.open_dataset(path_out)
            return f"Skipped {year}: output exists"
        except:
            pass

    # Load dataset
    ds = xr.open_dataset(path_in, chunks={"valid_time": 100})
    fine_lat = ds["latitude"].values
    fine_lon = ds["longitude"].values

    # Interpolate ERA5 -> EC-Earth3 grid (coarse)
    ds = ds.interp(
        latitude=coarse_lat,
        longitude=coarse_lon,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # Interpolate back to ERA5 grid (fine)
    ds = ds.interp(
        latitude=fine_lat, longitude=fine_lon, method="linear", kwargs={"fill_value": "extrapolate"}
    )

    # Save reinterpolated dataset
    ds.to_netcdf(path_out, encoding=encoding)

    return f"Processed {year}"


# --- Multiprocessing execution ---
if __name__ == "__main__":
    with Pool(1) as pool:
        results = list(tqdm(pool.imap(process_year, years), total=len(years)))
