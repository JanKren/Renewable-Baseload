# -*- coding: utf-8 -*-
"""
Download Wind Energy Data from ENTSO-E (2015-2024)

Downloads wind generation data for all European countries from 2015 to 2024.
Data is downloaded in quarterly chunks to avoid API timeouts.

Authors: Zajec & Kren, Jozef Stefan Institute
"""

from entsoe import EntsoePandasClient
import pandas as pd
import os
import time
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

API_KEY = "59b97bf1-8670-43e6-bcdc-d4f09288c112"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")

START_YEAR = 2015
END_YEAR = 2024

# Main countries (avoiding sub-zones to speed up download)
COUNTRIES = {
    'DE': 'Germany',
    'ES': 'Spain',
    'FR': 'France',
    'GB': 'United Kingdom',
    'SE': 'Sweden',
    'PL': 'Poland',
    'NL': 'Netherlands',
    'PT': 'Portugal',
    'DK': 'Denmark',
    'IE': 'Ireland',
    'BE': 'Belgium',
    'AT': 'Austria',
    'GR': 'Greece',
    'RO': 'Romania',
    'FI': 'Finland',
    'NO': 'Norway',
    'BG': 'Bulgaria',
    'HU': 'Hungary',
    'CZ': 'Czech Republic',
    'HR': 'Croatia',
    'LT': 'Lithuania',
    'EE': 'Estonia',
    'LV': 'Latvia',
    'SI': 'Slovenia',
    'SK': 'Slovakia',
    'LU': 'Luxembourg',
    'CH': 'Switzerland',
    'RS': 'Serbia',
    'IT': 'Italy',
}

API_DELAY = 2  # seconds between requests


def get_quarters(year):
    """Generate quarterly date ranges for a year."""
    quarters = [
        (f'{year}0101', f'{year}0401'),
        (f'{year}0401', f'{year}0701'),
        (f'{year}0701', f'{year}1001'),
        (f'{year}1001', f'{year+1}0101'),
    ]
    return quarters


def download_wind_data(client, country_code, start_str, end_str):
    """Download wind generation data for a specific period."""
    start = pd.Timestamp(start_str, tz='Europe/Brussels')
    end = pd.Timestamp(end_str, tz='Europe/Brussels')

    try:
        df = client.query_generation(country_code, start=start, end=end, psr_type=None)

        if df is None or df.empty:
            return None

        # Extract wind columns from MultiIndex columns
        wind_data = {}

        for col in df.columns:
            if isinstance(col, tuple):
                col_name = col[0].lower() if len(col) > 0 else ''
                col_type = col[1].lower() if len(col) > 1 else ''
            else:
                col_name = str(col).lower()
                col_type = 'actual'

            if 'wind' in col_name and 'actual' in col_type and 'consumption' not in col_type:
                if 'offshore' in col_name:
                    wind_data['Wind Offshore'] = df[col]
                elif 'onshore' in col_name:
                    wind_data['Wind Onshore'] = df[col]

        if not wind_data:
            return None

        result = pd.DataFrame(wind_data)
        return result

    except Exception as e:
        return None


def download_country(client, code, name):
    """Download all data for one country."""
    print(f"\n{code} - {name}")

    all_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        for q_num, (start_str, end_str) in enumerate(get_quarters(year), 1):
            print(f"  {year} Q{q_num}...", end=' ', flush=True)

            df = download_wind_data(client, code, start_str, end_str)

            if df is not None and not df.empty:
                all_data.append(df)
                print(f"OK ({len(df)})")
            else:
                print("No data")

            time.sleep(API_DELAY)

    if all_data:
        combined = pd.concat(all_data)
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        combined['Wind Total'] = combined.fillna(0).sum(axis=1)

        output_file = os.path.join(OUTPUT_DIR, f"{code}_wind_2015_2024.csv")
        combined.to_csv(output_file)
        print(f"  -> Saved: {len(combined)} rows to {output_file}")
        return True
    else:
        print(f"  -> No wind data for {code}")
        return False


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = EntsoePandasClient(api_key=API_KEY)

    print("=" * 60)
    print(f"ENTSO-E Wind Data Download: {START_YEAR}-{END_YEAR}")
    print(f"Countries: {len(COUNTRIES)}")
    print("=" * 60)

    successful = 0
    for i, (code, name) in enumerate(COUNTRIES.items(), 1):
        print(f"\n[{i}/{len(COUNTRIES)}]", end='')
        if download_country(client, code, name):
            successful += 1

    print("\n" + "=" * 60)
    print(f"Complete! Downloaded {successful}/{len(COUNTRIES)} countries")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"Finished at {datetime.now()}")
