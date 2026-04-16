import argparse
import calendar
from pathlib import Path

import cdsapi

TARGET = Path.home() / "ml-ds_data" / "CARRA2"


def download_data(years):
    """
    Download reanalysis data for specified year(s), grouped by month.

    Args:
        years: List of years (as strings) or a single year (int or string)
    """
    # Convert years to list of strings if needed
    if isinstance(years, (int, str)):  # noqa: SIM108
        years = [str(years)]
    else:
        years = [str(year) for year in years]

    dataset = "reanalysis-pan-carra"
    client = cdsapi.Client()

    # Download each year and month as a single request
    for year in years:
        for month_int in range(1, 13):
            month = f"{month_int:02d}"
            _, last_day = calendar.monthrange(int(year), month_int)
            days = [f"{day_int:02d}" for day_int in range(1, last_day + 1)]
            output_file = TARGET / year / f"CARRA{year}{month}.grib"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            if output_file.exists():
                print(f"Skipping existing file for {year}-{month}: {output_file}")
                continue

            request = {
                "level_type": "single_levels",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                    "land_sea_mask",
                    "orography",
                    "surface_roughness",
                ],
                "product_type": "analysis",
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
                "day": days,
                "area": [82, -10, 50, 45],
                "data_format": "grib",
            }

            print(f"Downloading data for {year}-{month} to {output_file}")
            client.retrieve(dataset, request).download(str(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download reanalysis data for specific years")
    parser.add_argument(
        "years", nargs="+", help="Year(s) to download (e.g., 1986 or 1986 1987 1988)"
    )

    args = parser.parse_args()
    download_data(args.years)
