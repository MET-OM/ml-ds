import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cdsapi

TARGET = Path.home() / "ml-ds_data" / "ERA5"

SINGLE_LEVELS_CONFIG = {
    "dataset": "reanalysis-era5-single-levels",
    "output_prefix": "ERA5SL",
    "request": {
        "product_type": ["reanalysis"],
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "boundary_layer_height",
            "surface_net_solar_radiation",
            "surface_net_thermal_radiation",
            "total_cloud_cover",
            "land_sea_mask",
            "sea_ice_cover",
            "surface_roughness",
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [90, -180, 40, 180],
    },
}

PRESSURE_LEVELS_CONFIG = {
    "dataset": "reanalysis-era5-pressure-levels",
    "output_prefix": "ERA5PL",
    "request": {
        "product_type": "reanalysis",
        "variable": [
            "temperature",
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_level": ["925", "850"],
        "format": "grib",
        "download_format": "unarchived",
        "area": [90, -180, 40, 180],
    },
}

DATASET_CONFIGS = {
    "single-levels": SINGLE_LEVELS_CONFIG,
    "pressure-levels": PRESSURE_LEVELS_CONFIG,
}


def _download_dataset(years, dataset_type):
    """
    Download one ERA5 dataset type for specified year(s).

    Args:
        years: List of years (as strings) or a single year (int or string)
        dataset_type: Dataset type to download, one of "single-levels" or "pressure-levels"
    """
    if isinstance(years, (int, str)):  # noqa: SIM108
        years = [str(years)]
    else:
        years = [str(year) for year in years]

    config = DATASET_CONFIGS[dataset_type]
    dataset = config["dataset"]
    output_prefix = config["output_prefix"]
    client = cdsapi.Client()

    for year in years:
        for month_int in range(1, 13):
            month = f"{month_int:02d}"
            _, last_day = calendar.monthrange(int(year), month_int)
            days = [f"{day_int:02d}" for day_int in range(1, last_day + 1)]
            output_file = TARGET / year / dataset_type / f"{output_prefix}{year}{month}.grib"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.exists():
                print(f"Skipping existing file for {year}-{month}: {output_file}")
                continue

            request = {
                **config["request"],
                "time": [
                    "00:00",
                    "06:00",
                    "12:00",
                    "18:00",
                ],
                "year": [year],
                "month": [month],
                "day": days,
            }

            print(f"Downloading {dataset_type} ERA5 for {year}-{month} to {output_file}")
            client.retrieve(dataset, request).download(str(output_file))


def download_data(years, dataset_type="single-levels"):
    """
    Download ERA5 reanalysis data for specified year(s).

    Args:
        years: List of years (as strings) or a single year (int or string)
        dataset_type: Dataset type to download, one of "single-levels", "pressure-levels", or "both"
    """
    if dataset_type == "both":
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_download_dataset, years, "single-levels"),
                executor.submit(_download_dataset, years, "pressure-levels"),
            ]
            for future in futures:
                future.result()
        return

    _download_dataset(years, dataset_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 reanalysis data for specific years")
    parser.add_argument(
        "--dataset",
        default="both",
        choices=(*DATASET_CONFIGS.keys(), "both"),
        help="Dataset to download: single-levels, pressure-levels, or both",
    )

    parser.add_argument(
        "years",
        nargs="+",
        help="Year(s) to download (e.g., 1986 or 1986 1987 1988)",
    )

    args = parser.parse_args()
    download_data(args.years, dataset_type=args.dataset)
