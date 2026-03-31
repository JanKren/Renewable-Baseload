# -*- coding: utf-8 -*-
"""
Download UK Wind Data from Elexon BMRS API (2021-2024)

Supplements ENTSO-E data which ends in June 2021 due to Brexit.

Authors: Zajec & Kren
"""

import pandas as pd
from elexonpy.api_client import ApiClient
from elexonpy.api.datasets_api import DatasetsApi
from datetime import datetime, timedelta
import time
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "GB_elexon_2021_2024.csv")

START_DATE = datetime(2021, 6, 15)
END_DATE = datetime(2024, 12, 31)


def download_uk_wind():
    """Download UK wind generation data from Elexon."""

    client = ApiClient()
    datasets_api = DatasetsApi(client)

    all_data = []
    current = START_DATE

    print("=" * 60)
    print("UK Wind Data Download from Elexon BMRS API")
    print("=" * 60)

    while current <= END_DATE:
        # Download one week at a time
        week_end = min(current + timedelta(days=7), END_DATE + timedelta(days=1))

        print(f"Downloading {current.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}...", end=' ', flush=True)

        try:
            result = datasets_api.datasets_agws_get(
                publish_date_time_from=current.strftime("%Y-%m-%dT00:00:00Z"),
                publish_date_time_to=week_end.strftime("%Y-%m-%dT00:00:00Z"),
                format="json"
            )

            if result.data:
                # Convert to list of dicts
                for record in result.data:
                    all_data.append({
                        'datetime': record.start_time,
                        'settlement_period': record.settlement_period,
                        'psr_type': record.psr_type,
                        'quantity': record.quantity
                    })
                print(f"OK ({len(result.data)} records)")
            else:
                print("No data")

        except Exception as e:
            print(f"Error: {e}")

        current = week_end
        time.sleep(0.5)  # Rate limiting

    if not all_data:
        print("No data downloaded!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Pivot to get Wind Onshore and Wind Offshore columns
    df_pivot = df.pivot_table(
        index='datetime',
        columns='psr_type',
        values='quantity',
        aggfunc='first'
    ).reset_index()

    df_pivot = df_pivot.set_index('datetime')

    # Rename columns
    col_mapping = {
        'Wind Onshore': 'Wind Onshore',
        'Wind Offshore': 'Wind Offshore'
    }
    df_pivot = df_pivot.rename(columns=col_mapping)

    # Calculate total wind
    wind_cols = [c for c in df_pivot.columns if 'Wind' in c]
    df_pivot['Wind Total'] = df_pivot[wind_cols].fillna(0).sum(axis=1)

    # Keep only wind columns
    df_pivot = df_pivot[[c for c in df_pivot.columns if 'Wind' in c or 'Solar' not in c]]

    # Sort by datetime
    df_pivot = df_pivot.sort_index()

    # Save
    df_pivot.to_csv(OUTPUT_FILE)

    print(f"\n" + "=" * 60)
    print(f"Saved: {len(df_pivot)} rows to {OUTPUT_FILE}")
    print(f"Date range: {df_pivot.index.min()} to {df_pivot.index.max()}")
    print(f"Wind production range: {df_pivot['Wind Total'].min():.0f} - {df_pivot['Wind Total'].max():.0f} MW")
    print("=" * 60)

    return df_pivot


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    download_uk_wind()
    print(f"Finished at {datetime.now()}")
