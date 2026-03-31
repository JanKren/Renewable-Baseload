#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERA5 Reanalysis Validation

Compares wind production correlations derived from ERA5 reanalysis data
with actual production data from ENTSO-E.

This addresses the field's most-acknowledged gap: validating reanalysis-based
findings with actual production data.

Steps:
1. Download ERA5 100m u/v wind components for country centroids
2. Compute wind speed and apply standard power curve
3. Calculate pairwise correlations from ERA5 capacity factors
4. Compare with actual production correlations
5. Quantify systematic biases

Requirements:
- CDS API credentials in ~/.cdsapirc or as environment variables
- Internet access for ERA5 download

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

try:
    import cdsapi
    import xarray as xr
    CDS_AVAILABLE = True
except ImportError:
    CDS_AVAILABLE = False
    print("Warning: cdsapi or xarray not available. Install with: pip install cdsapi xarray netCDF4")

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

# Country centroids (approximate lat, lon)
COUNTRY_CENTROIDS = {
    'AT': (47.5, 14.6),   # Austria
    'BE': (50.5, 4.5),    # Belgium
    'BG': (42.7, 25.5),   # Bulgaria
    'CH': (46.8, 8.2),    # Switzerland
    'CZ': (49.8, 15.5),   # Czech Republic
    'DE': (51.2, 10.5),   # Germany
    'DK': (56.3, 9.5),    # Denmark
    'EE': (58.6, 25.0),   # Estonia
    'ES': (40.5, -3.7),   # Spain
    'FI': (61.9, 25.7),   # Finland
    'FR': (46.2, 2.2),    # France
    'GB': (52.4, -1.5),   # United Kingdom
    'GR': (39.1, 21.8),   # Greece
    'HR': (45.1, 15.2),   # Croatia
    'HU': (47.2, 19.5),   # Hungary
    'IE': (53.4, -8.2),   # Ireland
    'IT': (41.9, 12.6),   # Italy
    'LT': (55.2, 23.9),   # Lithuania
    'LU': (49.8, 6.1),    # Luxembourg
    'LV': (56.9, 24.6),   # Latvia
    'NL': (52.1, 5.3),    # Netherlands
    'NO': (60.5, 8.5),    # Norway
    'PL': (51.9, 19.1),   # Poland
    'PT': (39.4, -8.2),   # Portugal
    'RO': (45.9, 25.0),   # Romania
    'RS': (44.0, 21.0),   # Serbia
    'SE': (60.1, 18.6),   # Sweden
    'SI': (46.2, 14.9),   # Slovenia
    'SK': (48.7, 19.7),   # Slovakia
}

# IEC Class II power curve (simplified)
# Wind speed (m/s) -> Capacity factor
def wind_to_capacity_factor(wind_speed):
    """
    Convert wind speed to capacity factor using simplified IEC Class II curve.

    Cut-in: 3 m/s
    Rated: 12 m/s
    Cut-out: 25 m/s
    """
    cf = np.zeros_like(wind_speed)

    # Below cut-in
    mask_low = wind_speed < 3
    cf[mask_low] = 0

    # Cubic region (cut-in to rated)
    mask_mid = (wind_speed >= 3) & (wind_speed < 12)
    cf[mask_mid] = ((wind_speed[mask_mid] - 3) / (12 - 3)) ** 3

    # Rated region
    mask_rated = (wind_speed >= 12) & (wind_speed <= 25)
    cf[mask_rated] = 1.0

    # Above cut-out
    mask_high = wind_speed > 25
    cf[mask_high] = 0

    return cf


# Nature Communications style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
})

COLORS = {
    'primary': '#2166ac',
    'secondary': '#b2182b',
    'highlight': '#5aae61',
    'neutral': '#878787',
}


def download_era5_for_year(year, countries, output_dir):
    """
    Download ERA5 100m wind data for specified countries and year.
    """
    if not CDS_AVAILABLE:
        raise RuntimeError("cdsapi not available")

    output_file = os.path.join(output_dir, f"era5_wind100m_{year}.nc")

    if os.path.exists(output_file):
        print(f"  ERA5 data for {year} already exists")
        return output_file

    # Get bounding box for all countries
    lats = [COUNTRY_CENTROIDS[c][0] for c in countries]
    lons = [COUNTRY_CENTROIDS[c][1] for c in countries]

    north = max(lats) + 2
    south = min(lats) - 2
    east = max(lons) + 2
    west = min(lons) - 2

    print(f"  Downloading ERA5 for {year} (area: {north}N/{west}W to {south}S/{east}E)...")

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '100m_u_component_of_wind',
                '100m_v_component_of_wind',
            ],
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
    """
    Extract wind speeds at country centroids from ERA5 netCDF file.
    """
    ds = xr.open_dataset(nc_file)

    # Get variables (may be named differently)
    u_var = 'u100' if 'u100' in ds else 'u100m' if 'u100m' in ds else list(ds.data_vars)[0]
    v_var = 'v100' if 'v100' in ds else 'v100m' if 'v100m' in ds else list(ds.data_vars)[1]

    # Get time variable (newer ERA5 uses 'valid_time' instead of 'time')
    time_var = 'valid_time' if 'valid_time' in ds else 'time'
    print(f"  Using time variable: {time_var}")

    country_data = {}

    for country, (lat, lon) in COUNTRY_CENTROIDS.items():
        if country not in countries:
            continue

        # Extract nearest grid point
        try:
            u = ds[u_var].sel(latitude=lat, longitude=lon, method='nearest')
            v = ds[v_var].sel(latitude=lat, longitude=lon, method='nearest')

            # Compute wind speed
            wind_speed = np.sqrt(u.values**2 + v.values**2)

            # Create time index
            time_idx = pd.to_datetime(ds[time_var].values)

            country_data[country] = pd.Series(wind_speed, index=time_idx)

        except Exception as e:
            print(f"    Error extracting {country}: {e}")

    ds.close()
    return country_data


def load_actual_production(data_dir):
    """Load actual production data from ENTSO-E."""
    country_data = {}
    files = glob.glob(os.path.join(data_dir, "*_wind_2015_2024.csv"))

    for filepath in files:
        filename = os.path.basename(filepath)
        country_code = filename.split('_')[0]

        try:
            df = pd.read_csv(filepath)
            date_col = df.columns[0]
            df['datetime'] = pd.to_datetime(df[date_col], utc=True)
            df = df.set_index('datetime')
            df.index = df.index.tz_localize(None)  # Remove timezone for comparison
            df = df.drop(columns=[date_col], errors='ignore')

            if 'Wind Total' in df.columns:
                df['wind'] = df['Wind Total']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df['wind'] = df[numeric_cols].fillna(0).sum(axis=1)

            # Normalize to capacity factor (approximate)
            max_val = df['wind'].quantile(0.99)
            if max_val > 0:
                df['cf'] = df['wind'] / max_val
                df['cf'] = df['cf'].clip(0, 1)
            else:
                df['cf'] = 0

            country_data[country_code] = df[['wind', 'cf']]

        except Exception as e:
            print(f"  Error loading {country_code}: {e}")

    return country_data


def compute_correlations(data_dict, var='cf'):
    """Compute pairwise correlations from data dictionary."""
    countries = sorted(data_dict.keys())
    n = len(countries)

    corr_matrix = pd.DataFrame(np.nan, index=countries, columns=countries)

    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i >= j:
                continue

            try:
                s1 = data_dict[c1][var] if isinstance(data_dict[c1], pd.DataFrame) and var in data_dict[c1] else data_dict[c1]
                s2 = data_dict[c2][var] if isinstance(data_dict[c2], pd.DataFrame) and var in data_dict[c2] else data_dict[c2]

                if isinstance(s1, pd.DataFrame):
                    s1 = s1[var] if var in s1.columns else s1.iloc[:, 0]
                if isinstance(s2, pd.DataFrame):
                    s2 = s2[var] if var in s2.columns else s2.iloc[:, 0]

                # Resample to hourly and align
                s1_h = s1.resample('h').mean()
                s2_h = s2.resample('h').mean()

                # Get common index and drop NaN together
                common_idx = s1_h.index.intersection(s2_h.index)

                if len(common_idx) > 1000:
                    df_pair = pd.DataFrame({'s1': s1_h.loc[common_idx], 's2': s2_h.loc[common_idx]}).dropna()
                    if len(df_pair) > 1000:
                        r = df_pair['s1'].corr(df_pair['s2'])
                        corr_matrix.loc[c1, c2] = r
                        corr_matrix.loc[c2, c1] = r
            except Exception as e:
                print(f"    Error computing correlation for {c1}-{c2}: {e}")

    # Set diagonal to 1.0
    for c in countries:
        corr_matrix.loc[c, c] = 1.0

    return corr_matrix


def compare_correlations(actual_corr, era5_corr):
    """Compare actual vs ERA5 correlations."""
    countries = sorted(set(actual_corr.index) & set(era5_corr.index))

    comparisons = []

    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i >= j:
                continue

            r_actual = actual_corr.loc[c1, c2]
            r_era5 = era5_corr.loc[c1, c2]

            if pd.notna(r_actual) and pd.notna(r_era5):
                comparisons.append({
                    'country1': c1,
                    'country2': c2,
                    'r_actual': r_actual,
                    'r_era5': r_era5,
                    'bias': r_era5 - r_actual,
                    'abs_bias': abs(r_era5 - r_actual)
                })

    return pd.DataFrame(comparisons)


def create_era5_validation_figure(comparison_df, actual_corr, era5_corr, output_dir):
    """Create ERA5 validation figure."""

    fig = plt.figure(figsize=(7.08, 5.5))

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.10, right=0.96, top=0.93, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel A: Scatter plot - ERA5 vs Actual correlations
    ax_a.scatter(comparison_df['r_actual'], comparison_df['r_era5'],
                 alpha=0.5, s=20, c=COLORS['primary'], edgecolors='white', linewidths=0.3)

    # 1:1 line
    lims = [min(ax_a.get_xlim()[0], ax_a.get_ylim()[0]),
            max(ax_a.get_xlim()[1], ax_a.get_ylim()[1])]
    ax_a.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=1.5, label='1:1 line')

    # Fit line
    z = np.polyfit(comparison_df['r_actual'], comparison_df['r_era5'], 1)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax_a.plot(x_line, np.poly1d(z)(x_line), '-', color=COLORS['highlight'], linewidth=1.5,
              label=f'Fit: y = {z[0]:.2f}x + {z[1]:.2f}')

    ax_a.set_xlabel('Actual Production Correlation')
    ax_a.set_ylabel('ERA5 Correlation')
    ax_a.set_title('a) ERA5 vs Actual Correlations', loc='left', fontweight='bold')
    ax_a.legend(loc='lower right', frameon=False, fontsize=6)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Add R² and RMSE
    r2 = np.corrcoef(comparison_df['r_actual'], comparison_df['r_era5'])[0, 1]**2
    rmse = np.sqrt(np.mean(comparison_df['bias']**2))
    ax_a.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}',
              transform=ax_a.transAxes, ha='left', va='top', fontsize=7,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    # Panel B: Bias distribution
    ax_b.hist(comparison_df['bias'], bins=30, alpha=0.7, color=COLORS['primary'],
              edgecolor='white', linewidth=0.5, density=True)
    ax_b.axvline(0, color=COLORS['secondary'], linestyle='--', linewidth=2)
    ax_b.axvline(comparison_df['bias'].mean(), color=COLORS['highlight'], linestyle='-', linewidth=2,
                 label=f"Mean bias: {comparison_df['bias'].mean():.3f}")

    ax_b.set_xlabel('Correlation Bias (ERA5 - Actual)')
    ax_b.set_ylabel('Density')
    ax_b.set_title('b) Bias Distribution', loc='left', fontweight='bold')
    ax_b.legend(loc='upper right', frameon=False, fontsize=6)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Panel C: Bias vs distance (approximate)
    # Use absolute correlation as proxy for distance
    ax_c.scatter(comparison_df['r_actual'], comparison_df['bias'],
                 alpha=0.5, s=20, c=COLORS['primary'], edgecolors='white', linewidths=0.3)
    ax_c.axhline(0, color=COLORS['secondary'], linestyle='--', linewidth=1)

    ax_c.set_xlabel('Actual Correlation')
    ax_c.set_ylabel('Bias (ERA5 - Actual)')
    ax_c.set_title('c) Bias vs Correlation Strength', loc='left', fontweight='bold')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Panel D: Summary statistics
    ax_d.axis('off')

    stats_text = f"""ERA5 Validation Summary

    Pairs analyzed: {len(comparison_df)}

    Correlation Statistics:
    - R² (ERA5 vs Actual): {r2:.3f}
    - RMSE: {rmse:.3f}
    - Mean bias: {comparison_df['bias'].mean():.3f}
    - Std bias: {comparison_df['bias'].std():.3f}

    Bias Direction:
    - ERA5 overestimates: {(comparison_df['bias'] > 0).sum()} pairs ({(comparison_df['bias'] > 0).mean()*100:.1f}%)
    - ERA5 underestimates: {(comparison_df['bias'] < 0).sum()} pairs ({(comparison_df['bias'] < 0).mean()*100:.1f}%)

    Key Finding:
    ERA5-based correlations {'systematically overestimate' if comparison_df['bias'].mean() > 0 else 'systematically underestimate'}
    actual production correlations by {abs(comparison_df['bias'].mean()):.3f} on average.
    """

    ax_d.text(0.5, 0.5, stats_text, transform=ax_d.transAxes, ha='center', va='center',
              fontsize=8, fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange'))

    ax_d.set_title('d) Summary', loc='left', fontweight='bold')

    # Save
    output_path = os.path.join(output_dir, 'figure_era5_validation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")


def main():
    print("=" * 70)
    print("ERA5 REANALYSIS VALIDATION")
    print("=" * 70)

    if not CDS_AVAILABLE:
        print("\nERROR: cdsapi not available. Please install with:")
        print("  pip install cdsapi xarray netCDF4")
        print("\nAlso ensure you have CDS API credentials in ~/.cdsapirc:")
        print("  url: https://cds.climate.copernicus.eu/api")
        print("  key: YOUR_UID:YOUR_API_KEY")
        return

    # Check for CDS credentials
    cdsapirc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(cdsapirc):
        print(f"\nWARNING: CDS API credentials not found at {cdsapirc}")
        print("Please create this file with your Copernicus CDS credentials:")
        print("  url: https://cds.climate.copernicus.eu/api")
        print("  key: YOUR_UID:YOUR_API_KEY")
        print("\nGet your API key from: https://cds.climate.copernicus.eu/profile")
        return

    print("\nLoading actual production data...")
    actual_data = load_actual_production(DATA_DIR)
    print(f"Loaded {len(actual_data)} countries")

    countries = list(actual_data.keys())

    # Download ERA5 data (for one year as demo - extend for full validation)
    print("\nDownloading ERA5 data (this may take a while)...")

    # Start with 2023 as a recent complete year
    years_to_download = [2023]

    era5_data = {}

    for year in years_to_download:
        print(f"\nProcessing {year}...")
        try:
            nc_file = download_era5_for_year(year, countries, ERA5_DIR)
            year_data = extract_country_wind_speeds(nc_file, countries)

            # Convert wind speed to capacity factor
            for country, wind_series in year_data.items():
                cf_series = pd.Series(wind_to_capacity_factor(wind_series.values),
                                     index=wind_series.index)
                if country not in era5_data:
                    era5_data[country] = cf_series
                else:
                    era5_data[country] = pd.concat([era5_data[country], cf_series])

        except Exception as e:
            print(f"  Error processing {year}: {e}")

    if not era5_data:
        print("\nNo ERA5 data available. Please check CDS API credentials.")
        return

    print(f"\nERA5 data available for {len(era5_data)} countries")

    # Compute correlations
    print("\nComputing correlations...")

    # Filter actual data to matching time period
    actual_filtered = {}
    for country, df in actual_data.items():
        if country in era5_data:
            # Get overlapping period
            era5_idx = era5_data[country].index
            mask = (df.index >= era5_idx.min()) & (df.index <= era5_idx.max())
            if mask.sum() > 1000:
                actual_filtered[country] = df.loc[mask]

    print(f"  Computing actual correlations ({len(actual_filtered)} countries)...")
    actual_corr = compute_correlations(actual_filtered, var='cf')

    print(f"  Computing ERA5 correlations...")
    era5_corr = compute_correlations({c: pd.DataFrame({'cf': s}) for c, s in era5_data.items()}, var='cf')

    # Compare
    print("\nComparing correlations...")
    comparison_df = compare_correlations(actual_corr, era5_corr)

    print("\n" + "-" * 50)
    print("VALIDATION RESULTS")
    print("-" * 50)
    print(f"Pairs compared: {len(comparison_df)}")
    print(f"Mean bias (ERA5 - Actual): {comparison_df['bias'].mean():.4f}")
    print(f"Std bias: {comparison_df['bias'].std():.4f}")
    print(f"RMSE: {np.sqrt(np.mean(comparison_df['bias']**2)):.4f}")
    print(f"R² (ERA5 vs Actual): {np.corrcoef(comparison_df['r_actual'], comparison_df['r_era5'])[0,1]**2:.4f}")

    print("\nLargest biases:")
    print(comparison_df.nlargest(5, 'abs_bias')[['country1', 'country2', 'r_actual', 'r_era5', 'bias']])

    # Create figure
    print("\nCreating validation figure...")
    create_era5_validation_figure(comparison_df, actual_corr, era5_corr, OUTPUT_DIR)

    # Save results
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'era5_validation_comparison.csv'), index=False)
    actual_corr.to_csv(os.path.join(RESULTS_DIR, 'actual_correlations.csv'))
    era5_corr.to_csv(os.path.join(RESULTS_DIR, 'era5_correlations.csv'))
    print(f"Results saved to {RESULTS_DIR}/")

    print("\n" + "=" * 70)
    print("ERA5 VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
