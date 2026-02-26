from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
import xesmf as xe


def flatten_forecast_var(ds, varname):
    nt = ds.dims["time"]
    ns = ds.dims["step"]

    data = ds[varname].values.reshape(nt * ns, ds.dims["latitude"], ds.dims["longitude"])

    new_time = ds["valid_time"].values.reshape(nt * ns)

    ds2 = xr.Dataset(
        {varname: (("time", "latitude", "longitude"), data)},
        coords={"time": new_time, "latitude": ds.latitude, "longitude": ds.longitude},
    )

    ds2 = ds2.sortby("time")
    ds2 = ds2.sel(time=~ds2.get_index("time").duplicated())

    return ds2


ds_ec = xr.open_dataset(Path.home() / "ml-ds_data" / "EC-Earth3.grid.nc")

coarse_lon = ds_ec["lon"].values
# Shift longitudes to the range [-180, 180] - for era5
coarse_lon = ((coarse_lon + 180) % 360) - 180
coarse_lon = np.sort(coarse_lon)
coarse_lat = ds_ec["lat"].values

ds_era5_pl = xr.open_dataset(
    sorted((Path.home() / "ml-ds_data" / "ERA5" / "2011" / "pressure-levels").glob("*.grib"))[0]
)

ds_era5_sl = cfgrib.open_datasets(
    sorted((Path.home() / "ml-ds_data" / "ERA5" / "2011" / "single-levels").glob("*.grib"))[0]
)

era_vars = {
    "x_t925": ds_era5_pl["t"].sel(isobaricInhPa=925),
    "x_t850": ds_era5_pl["t"].sel(isobaricInhPa=850),
    "x_z925": ds_era5_pl["z"].sel(isobaricInhPa=925) / 9.80665,  # to geopotential height at 925 hPa
    "x_z850": ds_era5_pl["z"].sel(isobaricInhPa=850) / 9.80665,
    "x_u925": ds_era5_pl["u"].sel(isobaricInhPa=925),
    "x_u850": ds_era5_pl["u"].sel(isobaricInhPa=850),
    "x_v925": ds_era5_pl["v"].sel(isobaricInhPa=925),
    "x_v850": ds_era5_pl["v"].sel(isobaricInhPa=850),
    "x_t2m": ds_era5_sl[0]["t2m"],
    "x_u10": ds_era5_sl[0]["u10"],
    "x_v10": ds_era5_sl[0]["v10"],
    "x_msl": ds_era5_sl[0]["msl"],  # mean sea level pressure
    "x_blh": ds_era5_sl[0]["blh"],  # boundary layer height
    "x_tcc": ds_era5_sl[0]["tcc"],  # total cloud cover
    "x_siconc": ds_era5_sl[0]["siconc"],  # sea ice fraction
    "x_ssr": flatten_forecast_var(ds_era5_sl[1], "ssr")["ssr"] / 3600.0,
    "x_str": flatten_forecast_var(ds_era5_sl[1], "str")["str"] / 3600.0,
}
assert (
    era_vars["x_t925"].shape
    == era_vars["x_t850"].shape
    == era_vars["x_z925"].shape
    == era_vars["x_z850"].shape
    == era_vars["x_u925"].shape
    == era_vars["x_u850"].shape
    == era_vars["x_v925"].shape
    == era_vars["x_v850"].shape
    == era_vars["x_t2m"].shape
    == era_vars["x_u10"].shape
    == era_vars["x_v10"].shape
    == era_vars["x_msl"].shape
    == era_vars["x_blh"].shape
    == era_vars["x_tcc"].shape
    == era_vars["x_siconc"].shape
    == era_vars["x_ssr"].shape
    == era_vars["x_str"].shape
)

carra_path = sorted((Path.home() / "ml-ds_data" / "CARRA2" / "2011").glob("*.grib"))[0]
ds_carra2 = cfgrib.open_datasets(carra_path)

x_slice = slice(2100, 2500)
y_slice = slice(400, 1400)

carra_vars = {
    "x_lsm": next(ds for ds in ds_carra2 if "lsm" in ds)["lsm"].isel(time=0, x=x_slice, y=y_slice),
    "x_orog": next(ds for ds in ds_carra2 if "orog" in ds)["orog"].isel(
        time=0, x=x_slice, y=y_slice
    ),
    "y_t2m": next(ds for ds in ds_carra2 if "t2m" in ds)["t2m"].isel(x=x_slice, y=y_slice),
    "y_u10": next(ds for ds in ds_carra2 if "u10" in ds)["u10"].isel(x=x_slice, y=y_slice),
    "y_v10": next(ds for ds in ds_carra2 if "v10" in ds)["v10"].isel(x=x_slice, y=y_slice),
}

ds_carra2_var = carra_vars["y_t2m"]


# Regrid ERA5 coarse t2m -> CARRA curvilinear grid
def _pick_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return ds.coords[name]
        if name in ds:
            return ds[name]
    raise KeyError(f"None of {candidates} found in dataset")


carra_lon = _pick_coord(ds_carra2_var, ["longitude", "lon"])
carra_lat = _pick_coord(ds_carra2_var, ["latitude", "lat"])

# Curvilinear target grid can be provided as 2D lon/lat arrays
grid_out = xr.Dataset({"lon": carra_lon, "lat": carra_lat})
regridder = None


def regrid_era5_to_carra(var_era, regridder=None, time_chunk=1):
    # Interpolate ERA5 -> EC-Earth3 grid (coarse)
    var_era_coarse = var_era.interp(
        latitude=coarse_lat,
        longitude=coarse_lon,
        method="linear",
    )
    var_era_coarse = var_era_coarse.dropna(dim="latitude", how="all").dropna(
        dim="longitude", how="all"
    )

    # xESMF expects source grid names lon/lat
    var_era_coarse = var_era_coarse.rename({"longitude": "lon", "latitude": "lat"})

    # Use lazy dask chunks and lower precision to keep memory bounded
    time_dim = next((d for d in ("time", "valid_time") if d in var_era_coarse.dims), None)
    chunk_map = {}
    if time_dim is not None:
        chunk_map[time_dim] = time_chunk
    if "lat" in var_era_coarse.dims:
        chunk_map["lat"] = min(120, var_era_coarse.sizes["lat"])
    if "lon" in var_era_coarse.dims:
        chunk_map["lon"] = min(120, var_era_coarse.sizes["lon"])
    if chunk_map:
        var_era_coarse = var_era_coarse.chunk(chunk_map)
    if np.issubdtype(var_era_coarse.dtype, np.floating):
        var_era_coarse = var_era_coarse.astype(np.float32, copy=False)

    if regridder is None:
        regridder = xe.Regridder(
            var_era_coarse,
            grid_out,
            method="bilinear",
            periodic=False,
            reuse_weights=False,
        )

    regrid_kwargs = {"keep_attrs": True, "skipna": True}
    if time_dim is not None:
        regrid_kwargs["output_chunks"] = {time_dim: time_chunk}

    var_era5_on_carra = regridder(var_era_coarse, **regrid_kwargs)
    return var_era5_on_carra, regridder


for var_name, var_era in list(era_vars.items()):
    var_era5_on_carra, regridder = regrid_era5_to_carra(var_era, regridder)
    era_vars[var_name] = var_era5_on_carra.compute()
    print(f"{var_name}: {var_era5_on_carra.shape}")


merged_ds = xr.Dataset()
for source in (era_vars, carra_vars):
    for key, value in source.items():
        merged_ds[key] = value.astype(np.float32, copy=False)


out_dir = Path.home() / "ml-ds_data" / "input_data"
out_dir.mkdir(parents=True, exist_ok=True)

ds_to_save = merged_ds.chunk({"time": 16})

zarr_path = out_dir / "2011.zarr"
ds_to_save.to_zarr(zarr_path, mode="w")

print(f"Saved merged_ds to: {zarr_path}")
