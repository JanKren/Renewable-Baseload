#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 Multi-Year Validation

Compares wind production correlations derived from ERA5 reanalysis data
with actual production data across multiple years (2015-2023).

This addresses the reviewer's request for validation over the full period,
not just a single year.

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import cdsapi
import xarray as xr

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
ERA5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data_ERA5")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "ERA5")

os.makedirs(ERA5_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Years to validate (spread across the decade)
VALIDATION_YEARS = [2015, 2017, 2019, 2021, 2023]

# Country centroids (approximate lat, lon)
COUNTRY_CENTROIDS = {
    'AT': (47.5, 14.6), 'BE': (50.5, 4.5), 'BG': (42.7, 25.5),
    'CZ': (49.8, 15.5), 'DE': (51.2, 10.5), 'DK': (56.3, 9.5),
    'EE': (58.6, 25.0), 'ES': (40.5, -3.7), 'FI': (61.9, 25.7),
    'FR': (46.2, 2.2), 'GB': (52.4, -1.5), 'GR': (39.1, 21.8),
    'HR': (45.1, 15.2), 'HU': (47.2, 19.5), 'IE': (53.4, -8.2),
    'IT': (41.9, 12.6), 'LT': (55.2, 23.9), 'LV': (56.9, 24.6),
    'NL': (52.1, 5.3), 'NO': (60.5, 8.5), 'PL': (51.9, 19.1),
    'PT': (39.4, -8.2), 'RO': (45.9, 25.0), 'SE': (60.1, 18.6),
}

# Power curve
def wind_to_capacity_factor(wind_speed):
    """IEC Class II power curve: cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s."""
    cf = np.zeros_like(wind_speed)
    mask_mid = (wind_speed >= 3) & (wind_speed < 12)
    cf[mask_mid] = ((wind_speed[mask_mid] - 3) / 9) ** 3
    mask_rated = (wind_speed >= 12) & (wind_speed <= 25)
    cf[mask_rated] = 1.0
    return cf

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'figure.dpi': 300,
})

COLORS = {'primary': '#2166ac', 'secondary': '#b2182b', 'highlight': '#5aae61'}


def download_era5_for_year(year, countries):
    """Download ERA5 100m wind data for specified year."""
    output_file = os.path.join(ERA5_DIR, f"era5_wind100m_{year}.nc")

    if os.path.exists(output_file):
        print(f"  ERA5 data for {year} already exists")
        return output_file

    lats = [COUNTRY_CENTROIDS[c][0] for c in countries]
    lons = [COUNTRY_CENTROIDS[c][1] for c in countries]

    north, south = max(lats) + 2, min(lats) - 2
    east, west = max(lons) + 2, min(lons) - 2

    print(f"  Downloading ERA5 for {year}...")

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['100m_u_component_of_wind', '100m_v_component_of_wind'],
            'year': str(year),
            'month': [f'{m:02d}' for m in range(1, 13)],
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(24)],
            'area': [north, west, south, east],
            'format': 'netcdf',
        },
        output_file
    )
    print(f"  Downloaded: {output_file}")
    return output_file


def extract_country_wind_speeds(nc_file, countries):
    """Extract wind speeds at country centroids from ERA5 netCDF."""
    ds = xr.open_dataset(nc_file)

    # Handle variable naming differences
    u_var = 'u100' if 'u100' in ds else 'u100m' if 'u100m' in ds else list(ds.data_vars)[0]
    v_var = 'v100' if 'v100' in ds else 'v100m' if 'v100m' in ds else list(ds.data_vars)[1]
    time_var = 'valid_time' if 'valid_time' in ds else 'time'

    country_data = {}
    for country, (lat, lon) in COUNTRY_CENTROIDS.items():
        if country not in countries:
            continue
        try:
            u = ds[u_var].sel(latitude=lat, longitude=lon, method='nearest')
            v = ds[v_var].sel(latitude=lat, longitude=lon, method='nearest')
            wind_speed = np.sqrt(u.values**2 + v.values**2)
            time_idx = pd.to_datetime(ds[time_var].values)
            country_data[country] = pd.Series(wind_speed, index=time_idx)
        except Exception as e:
            print(f"    Error extracting {country}: {e}")

    ds.close()
    return country_data


def load_actual_production(data_dir, year=None):
    """Load actual production data from ENTSO-E."""
    country_data = {}
    files = glob.glob(os.path.join(data_dir, "*_wind_2015_2024.csv"))

    for filepath in files:
        country_code = os.path.basename(filepath).split('_')[0]
        try:
            df = pd.read_csv(filepath)
            date_col = df.columns[0]
            df['datetime'] = pd.to_datetime(df[date_col], utc=True)
            df = df.set_index('datetime')
            df.index = df.index.tz_localize(None)

            if 'Wind Total' in df.columns:
                df['wind'] = df['Wind Total']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df['wind'] = df[numeric_cols].fillna(0).sum(axis=1)

            # Filter to year if specified
            if year:
                df = df[df.index.year == year]

            # Normalize to capacity factor
            max_val = df['wind'].quantile(0.99)
            df['cf'] = (df['wind'] / max_val).clip(0, 1) if max_val > 0 else 0

            if len(df) > 1000:
                country_data[country_code] = df[['wind', 'cf']]
        except Exception as e:
            pass

    return country_data


def compute_pairwise_correlations(data_dict, var='cf'):
    """Compute pairwise correlations."""
    countries = sorted(data_dict.keys())
    pairs = []

    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i >= j:
                continue
            try:
                s1 = data_dict[c1][var] if isinstance(data_dict[c1], pd.DataFrame) else data_dict[c1]
                s2 = data_dict[c2][var] if isinstance(data_dict[c2], pd.DataFrame) else data_dict[c2]

                s1_h = s1.resample('h').mean()
                s2_h = s2.resample('h').mean()
                common_idx = s1_h.index.intersection(s2_h.index)

                if len(common_idx) > 500:
                    df_pair = pd.DataFrame({'s1': s1_h.loc[common_idx], 's2': s2_h.loc[common_idx]}).dropna()
                    if len(df_pair) > 500:
                        r = df_pair['s1'].corr(df_pair['s2'])
                        pairs.append({'c1': c1, 'c2': c2, 'r': r})
            except:
                pass

    return pd.DataFrame(pairs)


def analyze_year(year, countries):
    """Analyze ERA5 vs actual correlations for a single year."""
    print(f"\n{'='*60}")
    print(f"YEAR {year}")
    print('='*60)

    # Download/load ERA5 data
    nc_file = download_era5_for_year(year, countries)
    era5_winds = extract_country_wind_speeds(nc_file, countries)

    # Convert to capacity factors
    era5_cf = {}
    for country, wind_series in era5_winds.items():
        cf = wind_to_capacity_factor(wind_series.values)
        era5_cf[country] = pd.DataFrame({'cf': cf}, index=wind_series.index)

    # Load actual data for this year
    actual_data = load_actual_production(DATA_DIR, year=year)

    # Filter to common countries
    common_countries = set(era5_cf.keys()) & set(actual_data.keys())
    print(f"  Common countries: {len(common_countries)}")

    era5_filtered = {c: era5_cf[c] for c in common_countries}
    actual_filtered = {c: actual_data[c] for c in common_countries}

    # Compute correlations
    era5_corr = compute_pairwise_correlations(era5_filtered)
    actual_corr = compute_pairwise_correlations(actual_filtered)

    # Merge and compute bias
    merged = era5_corr.merge(actual_corr, on=['c1', 'c2'], suffixes=('_era5', '_actual'))
    merged['bias'] = merged['r_era5'] - merged['r_actual']

    print(f"  Pairs analyzed: {len(merged)}")
    print(f"  Mean bias: {merged['bias'].mean():.4f}")
    print(f"  RMSE: {np.sqrt((merged['bias']**2).mean()):.4f}")

    return merged, year


def create_multiyear_figure(all_results, output_dir):
    """Create multi-year validation figure."""

    fig = plt.figure(figsize=(7.08, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])  # Scatter all years
    ax_b = fig.add_subplot(gs[0, 1])  # Bias by year
    ax_c = fig.add_subplot(gs[1, 0])  # Bias distribution
    ax_d = fig.add_subplot(gs[1, 1])  # Summary stats

    # Combine all results
    all_data = pd.concat([df.assign(year=year) for df, year in all_results])

    # Panel A: Scatter plot - all years combined
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))
    for (df, year), color in zip(all_results, colors):
        ax_a.scatter(df['r_actual'], df['r_era5'], alpha=0.4, s=15,
                    label=str(year), color=color)

    lims = [-0.2, 1.0]
    ax_a.plot(lims, lims, '--', color='red', linewidth=1.5, label='1:1')
    ax_a.set_xlim(lims)
    ax_a.set_ylim(lims)
    ax_a.set_xlabel('Actual Production Correlation')
    ax_a.set_ylabel('ERA5 Correlation')
    ax_a.set_title('a) ERA5 vs Actual (all years)', fontweight='bold', loc='left')
    ax_a.legend(loc='lower right', fontsize=6, ncol=2)

    # Panel B: Bias by year
    years = [year for _, year in all_results]
    mean_biases = [df['bias'].mean() for df, _ in all_results]
    std_biases = [df['bias'].std() for df, _ in all_results]

    ax_b.errorbar(years, mean_biases, yerr=std_biases, fmt='o-',
                  color=COLORS['primary'], capsize=5, linewidth=2, markersize=8)
    ax_b.axhline(0, color='red', linestyle='--', linewidth=1)
    ax_b.axhline(np.mean(mean_biases), color=COLORS['highlight'], linestyle='-',
                 linewidth=2, label=f'Overall mean: {np.mean(mean_biases):.3f}')
    ax_b.set_xlabel('Year')
    ax_b.set_ylabel('Mean Bias (ERA5 - Actual)')
    ax_b.set_title('b) Bias Evolution Over Time', fontweight='bold', loc='left')
    ax_b.legend(loc='upper right', fontsize=7)

    # Panel C: Bias distribution (all years)
    ax_c.hist(all_data['bias'], bins=40, alpha=0.7, color=COLORS['primary'],
              edgecolor='white', density=True)
    ax_c.axvline(0, color='red', linestyle='--', linewidth=2)
    ax_c.axvline(all_data['bias'].mean(), color=COLORS['highlight'], linestyle='-',
                 linewidth=2, label=f"Mean: {all_data['bias'].mean():.3f}")
    ax_c.set_xlabel('Correlation Bias (ERA5 - Actual)')
    ax_c.set_ylabel('Density')
    ax_c.set_title('c) Bias Distribution (pooled)', fontweight='bold', loc='left')
    ax_c.legend(loc='upper right', fontsize=7)

    # Panel D: Summary statistics table
    ax_d.axis('off')

    # Compute overall statistics
    r2 = np.corrcoef(all_data['r_actual'], all_data['r_era5'])[0, 1]**2
    rmse = np.sqrt((all_data['bias']**2).mean())

    summary = f"""
    ERA5 MULTI-YEAR VALIDATION SUMMARY
    {'='*45}

    Years analyzed: {', '.join(map(str, years))}
    Total pair-years: {len(all_data):,}

    OVERALL STATISTICS:
    • R² (ERA5 vs Actual):     {r2:.3f}
    • RMSE:                    {rmse:.3f}
    • Mean bias:               {all_data['bias'].mean():.3f}
    • Std bias:                {all_data['bias'].std():.3f}

    BIAS BY YEAR:
"""
    for (df, year) in all_results:
        summary += f"    {year}: {df['bias'].mean():+.3f} ± {df['bias'].std():.3f}\n"

    summary += f"""
    KEY FINDING:
    ERA5 systematically {'under' if all_data['bias'].mean() < 0 else 'over'}estimates
    actual production correlations by {abs(all_data['bias'].mean()):.3f} on average.
    This bias is {'stable' if np.std(mean_biases) < 0.02 else 'variable'} across years.
    """

    ax_d.text(0.05, 0.95, summary, transform=ax_d.transAxes, fontsize=7,
              fontfamily='monospace', verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    # Save
    output_path = os.path.join(output_dir, 'figure_era5_validation_multiyear.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_path}")
    return all_data


def main():
    print("=" * 70)
    print("ERA5 MULTI-YEAR VALIDATION")
    print("=" * 70)
    print(f"Started at {datetime.now()}")

    countries = list(COUNTRY_CENTROIDS.keys())
    all_results = []

    for year in VALIDATION_YEARS:
        try:
            result, yr = analyze_year(year, countries)
            if len(result) > 50:  # Minimum pairs
                all_results.append((result, yr))
        except Exception as e:
            print(f"  Error for {year}: {e}")

    if len(all_results) < 2:
        print("\nInsufficient data for multi-year analysis")
        return

    # Create figure
    print("\n" + "=" * 70)
    print("CREATING MULTI-YEAR FIGURE")
    print("=" * 70)

    all_data = create_multiyear_figure(all_results, OUTPUT_DIR)

    # Save results
    all_data.to_csv(os.path.join(RESULTS_DIR, 'era5_validation_multiyear.csv'), index=False)

    # Summary statistics by year
    summary_df = pd.DataFrame([
        {'year': year,
         'n_pairs': len(df),
         'mean_bias': df['bias'].mean(),
         'std_bias': df['bias'].std(),
         'rmse': np.sqrt((df['bias']**2).mean()),
         'r2': np.corrcoef(df['r_actual'], df['r_era5'])[0,1]**2}
        for df, year in all_results
    ])
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'era5_validation_summary_by_year.csv'), index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Years analyzed: {[y for _, y in all_results]}")
    print(f"Total pair-years: {len(all_data)}")
    print(f"Overall mean bias: {all_data['bias'].mean():.4f}")
    print(f"Overall RMSE: {np.sqrt((all_data['bias']**2).mean()):.4f}")
    print(f"Overall R²: {np.corrcoef(all_data['r_actual'], all_data['r_era5'])[0,1]**2:.4f}")

    print(f"\nFinished at {datetime.now()}")


if __name__ == "__main__":
    main()
