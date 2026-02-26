import argparse
import re
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
import xesmf as xe
from numcodecs import Blosc

DATA_ROOT = Path.home() / "ml-ds_data"
ERA5_ROOT = DATA_ROOT / "ERA5"
CARRA_ROOT = DATA_ROOT / "CARRA2"
EC_GRID_FILE = DATA_ROOT / "EC-Earth3.grid.nc"
OUTPUT_ROOT = DATA_ROOT / "input_data"

X_SLICE = slice(2100, 2500)
Y_SLICE = slice(400, 1400)
TIME_CHUNK = 16
ZARR_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)


def flatten_forecast_var(ds, varname):
    nt = ds.sizes["time"]
    ns = ds.sizes["step"]
    data = ds[varname].values.reshape(nt * ns, ds.sizes["latitude"], ds.sizes["longitude"])
    new_time = ds["valid_time"].values.reshape(nt * ns)
    ds2 = xr.Dataset(
        {varname: (("time", "latitude", "longitude"), data)},
        coords={"time": new_time, "latitude": ds.latitude, "longitude": ds.longitude},
    )
    ds2 = ds2.sortby("time")
    ds2 = ds2.sel(time=~ds2.get_index("time").duplicated())
    return ds2


def _pick_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords:
            return ds.coords[name]
        if name in ds:
            return ds[name]
    raise KeyError(f"None of {candidates} found in dataset")


def _extract_yyyymm(path):
    match = re.search(r"(\d{6})", path.name)
    if match is None:
        return None
    return match.group(1)


def _files_by_month(folder):
    mapping = {}
    for file_path in sorted(folder.glob("*.grib")):
        month_key = _extract_yyyymm(file_path)
        if month_key is not None:
            mapping[month_key] = file_path
    return mapping


def build_month_index(year):
    year_str = str(year)
    era_pl = _files_by_month(ERA5_ROOT / year_str / "pressure-levels")
    era_sl = _files_by_month(ERA5_ROOT / year_str / "single-levels")
    carra = _files_by_month(CARRA_ROOT / year_str)

    common_months = sorted(set(era_pl) & set(era_sl) & set(carra))
    if not common_months:
        raise FileNotFoundError(
            f"No matching monthly files found for year {year_str} in ERA5/CARRA folders"
        )

    return [
        {
            "month": month,
            "era5_pl": era_pl[month],
            "era5_sl": era_sl[month],
            "carra": carra[month],
        }
        for month in common_months
    ]


def get_coarse_grid():
    ds_ec = xr.open_dataset(EC_GRID_FILE)
    coarse_lon = ds_ec["lon"].values
    coarse_lon = ((coarse_lon + 180) % 360) - 180
    coarse_lon = np.sort(coarse_lon)
    coarse_lat = ds_ec["lat"].values
    return coarse_lat, coarse_lon


def _find_dataset_with_var(datasets, varname):
    return next(ds for ds in datasets if varname in ds)


def load_era_vars(era5_pl_file, era5_sl_file):
    ds_era5_pl = xr.open_dataset(era5_pl_file)
    ds_era5_sl = cfgrib.open_datasets(era5_sl_file)

    ds_sl_analysis = _find_dataset_with_var(ds_era5_sl, "t2m")
    ds_sl_fc = _find_dataset_with_var(ds_era5_sl, "ssr")

    return {
        "x_t925": ds_era5_pl["t"].sel(isobaricInhPa=925),
        "x_t850": ds_era5_pl["t"].sel(isobaricInhPa=850),
        "x_z925": ds_era5_pl["z"].sel(isobaricInhPa=925) / 9.80665,
        "x_z850": ds_era5_pl["z"].sel(isobaricInhPa=850) / 9.80665,
        "x_u925": ds_era5_pl["u"].sel(isobaricInhPa=925),
        "x_u850": ds_era5_pl["u"].sel(isobaricInhPa=850),
        "x_v925": ds_era5_pl["v"].sel(isobaricInhPa=925),
        "x_v850": ds_era5_pl["v"].sel(isobaricInhPa=850),
        "x_t2m": ds_sl_analysis["t2m"],
        "x_u10": ds_sl_analysis["u10"],
        "x_v10": ds_sl_analysis["v10"],
        "x_msl": ds_sl_analysis["msl"],
        "x_blh": ds_sl_analysis["blh"],
        "x_tcc": ds_sl_analysis["tcc"],
        "x_siconc": ds_sl_analysis["siconc"],
        "x_ssr": flatten_forecast_var(ds_sl_fc, "ssr")["ssr"] / 3600.0,
        "x_str": flatten_forecast_var(ds_sl_fc, "str")["str"] / 3600.0,
    }


def load_carra_vars(carra_file, include_static):
    ds_carra = cfgrib.open_datasets(carra_file)
    ds_t2m = _find_dataset_with_var(ds_carra, "t2m")
    ds_u10 = _find_dataset_with_var(ds_carra, "u10")
    ds_v10 = _find_dataset_with_var(ds_carra, "v10")

    carra_vars = {
        "y_t2m": ds_t2m["t2m"].isel(x=X_SLICE, y=Y_SLICE),
        "y_u10": ds_u10["u10"].isel(x=X_SLICE, y=Y_SLICE),
        "y_v10": ds_v10["v10"].isel(x=X_SLICE, y=Y_SLICE),
    }

    if include_static:
        ds_lsm = _find_dataset_with_var(ds_carra, "lsm")
        ds_orog = _find_dataset_with_var(ds_carra, "orog")
        carra_vars["x_lsm"] = ds_lsm["lsm"].isel(time=0, x=X_SLICE, y=Y_SLICE)
        carra_vars["x_orog"] = ds_orog["orog"].isel(time=0, x=X_SLICE, y=Y_SLICE)

    return carra_vars, carra_vars["y_t2m"]


def regrid_era5_to_carra(var_era, coarse_lat, coarse_lon, grid_out, regridder=None, time_chunk=1):
    var_era_coarse = var_era.interp(latitude=coarse_lat, longitude=coarse_lon, method="linear")
    var_era_coarse = var_era_coarse.dropna(dim="latitude", how="all").dropna(
        dim="longitude", how="all"
    )
    var_era_coarse = var_era_coarse.rename({"longitude": "lon", "latitude": "lat"})

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


def process_month(
    month_info,
    coarse_lat,
    coarse_lon,
    grid_out=None,
    regridder=None,
    include_static=False,
):
    era_vars = load_era_vars(month_info["era5_pl"], month_info["era5_sl"])
    carra_vars, ref_carra = load_carra_vars(month_info["carra"], include_static=include_static)

    if grid_out is None:
        carra_lon = _pick_coord(ref_carra, ["longitude", "lon"])
        carra_lat = _pick_coord(ref_carra, ["latitude", "lat"])
        grid_out = xr.Dataset({"lon": carra_lon, "lat": carra_lat})

    regridded_era = {}
    for var_name, var_era in era_vars.items():
        var_on_carra, regridder = regrid_era5_to_carra(
            var_era,
            coarse_lat=coarse_lat,
            coarse_lon=coarse_lon,
            grid_out=grid_out,
            regridder=regridder,
            time_chunk=1,
        )
        regridded_era[var_name] = var_on_carra  # .compute()

    merged_ds = xr.Dataset()
    for source in (regridded_era, carra_vars):
        for key, value in source.items():
            merged_ds[key] = value.astype(np.float32, copy=False)

    return merged_ds, grid_out, regridder


def write_month_to_zarr(merged_ds, zarr_path, first_month):
    ds_to_save = merged_ds.chunk({"time": TIME_CHUNK}) if "time" in merged_ds.dims else merged_ds
    encoding = {var_name: {"compressor": ZARR_COMPRESSOR} for var_name in ds_to_save.data_vars}

    if first_month:
        ds_to_save.to_zarr(
            zarr_path,
            mode="w",
            encoding=encoding,
            zarr_format=2,
            align_chunks=True,
        )
    else:
        try:
            ds_to_save.to_zarr(
                zarr_path,
                mode="a",
                append_dim="time",
                zarr_format=2,
                align_chunks=True,
            )
        except ValueError as exc:
            if "overlap multiple Dask chunks" not in str(exc):
                raise
            print(
                "Chunk alignment warning: detected Dask/Zarr chunk overlap during append; "
                "retrying with a safe time rechunk (time=1)"
            )
            ds_safe = ds_to_save.chunk({"time": 1})
            ds_safe.to_zarr(
                zarr_path,
                mode="a",
                append_dim="time",
                zarr_format=2,
                align_chunks=True,
            )


def prepare_year(year, output_dir):
    month_index = build_month_index(year)
    coarse_lat, coarse_lon = get_coarse_grid()

    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = output_dir / f"{year}.zarr"
    if zarr_path.exists():
        print(f"Overwriting existing output: {zarr_path}")

    grid_out = None
    regridder = None

    for idx, month_info in enumerate(month_index):
        include_static = idx == 0
        print(
            f"Processing {year}-{month_info['month'][-2:]}: ",
            (
                f"{month_info['era5_pl'].name}, "
                f"{month_info['era5_sl'].name}, "
                f"{month_info['carra'].name}"
            ),
        )
        merged_ds, grid_out, regridder = process_month(
            month_info,
            coarse_lat=coarse_lat,
            coarse_lon=coarse_lon,
            grid_out=grid_out,
            regridder=regridder,
            include_static=include_static,
        )
        write_month_to_zarr(merged_ds, zarr_path, first_month=(idx == 0))

    print(f"Saved yearly dataset to: {zarr_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare yearly ML input zarr by processing all monthly files in each year folder"
        )
    )
    parser.add_argument(
        "years",
        nargs="+",
        help="Year(s) to process (e.g. 2011 or 2011 2012)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output directory for yearly zarr files (default: {OUTPUT_ROOT})",
    )
    args = parser.parse_args()

    for year in args.years:
        prepare_year(year, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
