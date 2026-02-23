import argparse
import calendar
from pathlib import Path

import cdsapi

TARGET = Path.home() / "ml-ds_data" / "ERA5"


def download_data(years):
    """
    Download ERA5 reanalysis data for specified year(s).

    Args:
        years: List of years (as strings) or a single year (int or string)
    """
    if isinstance(years, (int, str)):  # noqa: SIM108
        years = [str(years)]
    else:
        years = [str(year) for year in years]

    TARGET.mkdir(parents=True, exist_ok=True)

    dataset = "reanalysis-era5-single-levels"
    client = cdsapi.Client()

    for year in years:
        for month_int in range(1, 13):
            month = f"{month_int:02d}"
            _, last_day = calendar.monthrange(int(year), month_int)
            for day_int in range(1, last_day + 1):
                day = f"{day_int:02d}"
                output_file = TARGET / f"ERA5{year}{month}{day}.nc"

                if output_file.exists():
                    print(f"Skipping existing file for {year}-{month}-{day}: {output_file}")
                    continue

                request = {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_dewpoint_temperature",
                        "2m_temperature",
                    ],
                    "time": [
                        "00:00",
                        "03:00",
                        "06:00",
                        "09:00",
                        "12:00",
                        "15:00",
                        "18:00",
                        "21:00",
                    ],
                    "year": [year],
                    "month": [month],
                    "day": [day],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    "area": [90, -180, 40, 180],
                }

                print(f"Downloading ERA5 for {year}-{month}-{day} to {output_file}")
                client.retrieve(dataset, request).download(str(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 reanalysis data for specific years")
    parser.add_argument(
        "years",
        nargs="+",
        help="Year(s) to download (e.g., 1986 or 1986 1987 1988)",
    )

    args = parser.parse_args()
    download_data(args.years)
