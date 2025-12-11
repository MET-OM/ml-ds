import xarray as xr
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

base_folder = "download/"
annual_folder = "original/"

file_list = np.sort([base_folder + f for f in os.listdir(base_folder)])

encoding = {
    "t2m": {
        'dtype': 'int16',
        'scale_factor': 0.01,
        'add_offset': 273.0,
        'zlib': True,
        'shuffle': True,
        'complevel': 1,
        'chunksizes': (1, 161, 181),
        '_FillValue': np.int16(-32768)
    },
    "d2m": {
        'dtype': 'int16',
        'scale_factor': 0.01,
        'add_offset': 273.0,
        'zlib': True,
        'shuffle': True,
        'complevel': 1,
        'chunksizes': (1, 161, 181),
        '_FillValue': np.int16(-32768)
    },
    "u10": { 
        'dtype': 'int16',
        'scale_factor': 0.01,
        'add_offset': 0.0,
        'zlib': True,
        'shuffle': False,
        'complevel': 1,
        'chunksizes': (1, 161, 181),
        '_FillValue': np.int16(-32768)
    },
    "v10": {
        'dtype': 'int16',
        'scale_factor': 0.01,
        'add_offset': 0.0,
        'zlib': True,
        'shuffle': False,
        'complevel': 1,
        'chunksizes': (1, 161, 181),
        '_FillValue': np.int16(-32768)
    }
}

years = np.arange(1940, 2026)

def process_year(year):
    path_spring = f"{base_folder}ERA5_{year}_spring.nc"
    path_autumn = f"{base_folder}ERA5_{year}_autumn.nc"
    path_annual = f"{annual_folder}ERA5_{year}.nc"

    if not (os.path.exists(path_spring) and os.path.exists(path_autumn)):
        return f"Skipped {year}: missing file"

    # Load datasets
    spring = xr.load_dataset(path_spring)
    autumn = xr.load_dataset(path_autumn)

    # Concatenate along the desired dimension
    annual = xr.concat([spring, autumn], dim="valid_time")

    # Save to NetCDF with encoding
    annual.to_netcdf(path_annual, encoding=encoding)
    return f"Processed {year}"

if __name__ == "__main__":
    # Use a progress bar with multiprocessing
    with Pool(processes=7) as pool:
        results = list(tqdm(pool.imap(process_year, years), total=len(years)))

    # Print summary
    for r in results:
        print(r)
