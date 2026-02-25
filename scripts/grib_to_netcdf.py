from __future__ import annotations

import argparse
from pathlib import Path

import cfgrib
import xarray as xr


def convert_grib_file(path_in: Path, path_out: Path, overwrite: bool) -> str:
	if path_out.exists() and not overwrite:
		return f"Skipped {path_in.name} (already exists)"

	parts = cfgrib.open_datasets(path_in, backend_kwargs={"indexpath": ""})
	if not parts:
		raise ValueError(f"No datasets found in GRIB file: {path_in}")

	if len(parts) == 1:
		ds = parts[0]
	else:
		ds = xr.merge(parts, compat="override", join="outer")

	try:
		ds.to_netcdf(path_out)
	finally:
		ds.close()
		for part in parts:
			part.close()

	return f"Converted {path_in.name} -> {path_out.name}"


def convert_directory(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	grib_files = sorted(input_dir.glob("*.grib"))

	if not grib_files:
		raise FileNotFoundError(f"No .grib files found in: {input_dir}")

	for grib_file in grib_files:
		out_file = output_dir / f"{grib_file.stem}.nc"
		result = convert_grib_file(grib_file, out_file, overwrite)
		print(result)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert all .grib files in a directory to NetCDF")
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="Directory containing .grib files (default: this script directory)",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory to write .nc files (default: same as input-dir)",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing .nc files",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_dir = args.input_dir.resolve()
	output_dir = args.output_dir.resolve() if args.output_dir is not None else input_dir
	convert_directory(input_dir, output_dir, args.overwrite)


if __name__ == "__main__":
	main()
