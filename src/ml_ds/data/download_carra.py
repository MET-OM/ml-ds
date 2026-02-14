import argparse
from pathlib import Path

import cdsapi

TARGET = Path.home() / "CARRA"


def download_data(years):
    """
    Download reanalysis data for specified year(s).
    
    Args:
        years: List of years (as strings) or a single year (int or string)
    """
    # Convert years to list of strings if needed
    if isinstance(years, (int, str)):  # noqa: SIM108
        years = [str(years)]
    else:
        years = [str(year) for year in years]
    
    # Create target directory if it doesn't exist
    TARGET.mkdir(parents=True, exist_ok=True)
    
    dataset = "reanalysis-pan-carra"
    client = cdsapi.Client()
    
    months = [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12"
    ]
    
    # Download each year and month separately
    for year in years:
        for month in months:
            output_file = TARGET / f"CARRA{year}{month}.nc"
            
            request = {
                "level_type": "single_levels",
                    "variable": [
                    "2m_temperature",
                    "land_sea_mask",
                    "orography"
                ],
                "product_type": "analysis",
                "time": [
                    "00:00", "03:00", "06:00",
                    "09:00", "12:00", "15:00",
                    "18:00", "21:00"
                ],
                "year": [year],
                "month": [month],
                "day": [
                    "01", "02", "03",
                    "04", "05", "06",
                    "07", "08", "09",
                    "10", "11", "12",
                    "13", "14", "15",
                    "16", "17", "18",
                    "19", "20", "21",
                    "22", "23", "24",
                    "25", "26", "27",
                    "28", "29", "30",
                    "31"
                ],
                "data_format": "netcdf"  # "grib"
            }
            
            print(f"Downloading data for {year}-{month} to {output_file}")
            client.retrieve(dataset, request).download(str(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download reanalysis data for specific years")
    parser.add_argument(
        "years",
        nargs="+",
        help="Year(s) to download (e.g., 1986 or 1986 1987 1988)"
    )
    
    args = parser.parse_args()
    download_data(args.years)
