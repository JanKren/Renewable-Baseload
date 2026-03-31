# -*- coding: utf-8 -*-
"""
Download UK Solar Data from Elexon BMRS API (2021-2024)

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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "GB_solar_elexon_2021_2024.csv")

# Start from where ENTSO-E data ends (mid-June 2021)
START_DATE = datetime(2021, 6, 15)
END_DATE = datetime(2024, 12, 31)


def download_uk_solar():
    """Download UK solar generation data from Elexon."""

    client = ApiClient()
    datasets_api = DatasetsApi(client)

    all_data = []
    current = START_DATE

    print("=" * 60)
    print("UK Solar Data Download from Elexon BMRS API")
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
                # Filter for Solar records only
                solar_count = 0
                for record in result.data:
                    if record.psr_type and 'Solar' in record.psr_type:
                        all_data.append({
                            'datetime': record.start_time,
                            'settlement_period': record.settlement_period,
                            'psr_type': record.psr_type,
                            'quantity': record.quantity
                        })
                        solar_count += 1
                print(f"OK ({solar_count} solar records)")
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

    # Pivot to get Solar column
    df_pivot = df.pivot_table(
        index='datetime',
        columns='psr_type',
        values='quantity',
        aggfunc='first'
    ).reset_index()

    df_pivot = df_pivot.set_index('datetime')

    # Calculate total solar (in case there are multiple solar types)
    solar_cols = [c for c in df_pivot.columns if 'Solar' in c]
    if len(solar_cols) == 1:
        df_pivot['Solar'] = df_pivot[solar_cols[0]]
    else:
        df_pivot['Solar'] = df_pivot[solar_cols].fillna(0).sum(axis=1)

    # Keep only the Solar column
    df_pivot = df_pivot[['Solar']]

    # Sort by datetime
    df_pivot = df_pivot.sort_index()

    # Save
    df_pivot.to_csv(OUTPUT_FILE)

    print(f"\n" + "=" * 60)
    print(f"Saved: {len(df_pivot)} rows to {OUTPUT_FILE}")
    print(f"Date range: {df_pivot.index.min()} to {df_pivot.index.max()}")
    print(f"Solar production range: {df_pivot['Solar'].min():.0f} - {df_pivot['Solar'].max():.0f} MW")
    print(f"Mean solar production: {df_pivot['Solar'].mean():.0f} MW")
    print("=" * 60)

    return df_pivot


def merge_with_entsoe():
    """Merge Elexon data with existing ENTSO-E GB solar data."""

    entsoe_file = os.path.join(OUTPUT_DIR, "GB_solar_2015_2024.csv")
    merged_file = os.path.join(OUTPUT_DIR, "GB_solar_complete_2015_2024.csv")

    if not os.path.exists(entsoe_file):
        print(f"ENTSO-E file not found: {entsoe_file}")
        print("Skipping merge - will use Elexon data only")
        return

    if not os.path.exists(OUTPUT_FILE):
        print(f"Elexon file not found: {OUTPUT_FILE}")
        print("Run download first")
        return

    print("\nMerging ENTSO-E and Elexon data...")

    # Load ENTSO-E data
    df_entsoe = pd.read_csv(entsoe_file, index_col=0, parse_dates=True)

    # Load Elexon data
    df_elexon = pd.read_csv(OUTPUT_FILE, index_col=0, parse_dates=True)

    # ENTSO-E data ends around mid-June 2021
    # Use ENTSO-E before 2021-06-15, Elexon after
    cutoff = pd.Timestamp('2021-06-15')

    df_entsoe_early = df_entsoe[df_entsoe.index < cutoff]
    df_elexon_late = df_elexon[df_elexon.index >= cutoff]

    # Rename columns to match
    if 'Solar' not in df_entsoe_early.columns and len(df_entsoe_early.columns) == 1:
        df_entsoe_early.columns = ['Solar']

    # Combine
    df_merged = pd.concat([df_entsoe_early[['Solar']], df_elexon_late[['Solar']]])
    df_merged = df_merged.sort_index()

    # Remove duplicates
    df_merged = df_merged[~df_merged.index.duplicated(keep='first')]

    # Save
    df_merged.to_csv(merged_file)

    print(f"Merged file saved: {merged_file}")
    print(f"Total rows: {len(df_merged)}")
    print(f"Date range: {df_merged.index.min()} to {df_merged.index.max()}")
    print(f"ENTSO-E rows: {len(df_entsoe_early)}, Elexon rows: {len(df_elexon_late)}")

    return df_merged


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    df = download_uk_solar()
    if df is not None:
        merge_with_entsoe()
    print(f"Finished at {datetime.now()}")
