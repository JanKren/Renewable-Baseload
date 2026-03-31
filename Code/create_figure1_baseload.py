# -*- coding: utf-8 -*-
"""
Figure 1: Wind Never Stops - Baseload Evolution

Publication-quality figure for Nature Communications showing:
- Panel A: All years hourly time series normalized by installed capacity (CF)
- Panel B: 10-year baseload evolution bar chart with trend
- Panel C: Capacity factor distribution histogram

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from scipy import stats
from scipy.special import gamma
import os
import glob
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "MultiYear")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# European installed wind capacity by year (GW) - from WindEurope statistics
INSTALLED_CAPACITY = {
    2015: 142,
    2016: 154,
    2017: 169,
    2018: 189,
    2019: 205,
    2020: 220,
    2021: 236,
    2022: 255,
    2023: 272,
    2024: 300,
}

# Nature Communications style settings
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
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# Color scheme (colorblind-friendly)
COLORS = {
    'primary': '#2166ac',      # Blue
    'secondary': '#b2182b',    # Red
    'highlight': '#d6604d',    # Light red
    'accent': '#4393c3',       # Light blue
    'neutral': '#878787',      # Gray
    'fill': '#92c5de',         # Very light blue
    'weibull': '#762a83',      # Purple for Weibull fit
}


def load_all_country_data(data_dir):
    """Load all country wind data files."""
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
            df = df.drop(columns=[date_col], errors='ignore')

            if 'Wind Total' in df.columns:
                df['wind'] = df['Wind Total']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df['wind'] = df[numeric_cols].fillna(0).sum(axis=1)

            country_data[country_code] = df[['wind']]

        except Exception as e:
            print(f"  Error loading {country_code}: {e}")

    return country_data


def get_aggregated_timeseries(country_data, year):
    """Get aggregated European wind production for a specific year."""
    year_data = {}

    for country, df in country_data.items():
        year_df = df[df.index.year == year]
        if len(year_df) > 100:
            hourly = year_df['wind'].resample('h').mean().dropna()
            if len(hourly) > 100:
                year_data[country] = hourly

    if len(year_data) < 3:
        return None

    # Find common timestamps
    common_index = None
    for country, series in year_data.items():
        if common_index is None:
            common_index = set(series.index)
        else:
            common_index = common_index.intersection(set(series.index))

    common_index = sorted(list(common_index))

    # Aggregate
    total = pd.Series(0.0, index=common_index)
    for country, series in year_data.items():
        total += series.loc[common_index].fillna(0)

    return total


def create_figure1(country_data, baseload_df):
    """Create Figure 1: Baseload Analysis multi-panel figure."""

    # Create figure: 180mm wide, ~140mm tall (Nature Comms format)
    fig = plt.figure(figsize=(7.08, 5.5))

    # Create grid with custom spacing
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[2, 1],
                          hspace=0.35, wspace=0.3,
                          left=0.08, right=0.96, top=0.94, bottom=0.08)

    # Panel A: All years time series normalized (spans full width on top)
    ax_a = fig.add_subplot(gs[0, :])

    # Panel B: 10-year evolution (bottom left)
    ax_b = fig.add_subplot(gs[1, 0])

    # Panel C: Capacity factor histogram (bottom right)
    ax_c = fig.add_subplot(gs[1, 1])

    # =========================================================================
    # Panel A: All Years Normalized by Installed Capacity (Capacity Factor)
    # =========================================================================
    years = list(range(2015, 2025))
    cmap = cm.get_cmap('viridis', len(years))

    all_cf_data = []

    for i, year in enumerate(years):
        ts = get_aggregated_timeseries(country_data, year)
        if ts is not None:
            # Convert to capacity factor (normalize by installed capacity)
            capacity_gw = INSTALLED_CAPACITY[year]
            cf = (ts / 1000) / capacity_gw  # Capacity factor (0-1)

            # Create day-of-year index for alignment
            doy = ts.index.dayofyear + (ts.index.hour / 24)

            # Plot with color gradient from old (light) to new (dark)
            alpha = 0.3 + 0.5 * (i / len(years))  # Increasing opacity
            ax_a.plot(doy, cf.values, linewidth=0.3, color=cmap(i), alpha=alpha,
                     label=str(year) if i % 2 == 0 else None)

            all_cf_data.append(cf.values)

    # Calculate and plot envelope (min/max across years)
    min_length = min(len(d) for d in all_cf_data)
    cf_array = np.array([d[:min_length] for d in all_cf_data])
    cf_mean = np.mean(cf_array, axis=0)
    cf_min = np.min(cf_array, axis=0)
    cf_max = np.max(cf_array, axis=0)

    doy_common = np.linspace(1, 365, min_length)
    ax_a.fill_between(doy_common, cf_min, cf_max, alpha=0.2, color=COLORS['neutral'],
                      label='Range (2015-2024)')

    # Mark overall minimum
    overall_min = np.min(cf_array)
    min_idx = np.unravel_index(np.argmin(cf_array), cf_array.shape)
    ax_a.axhline(overall_min, color=COLORS['secondary'], linestyle='--', linewidth=1,
                 alpha=0.7)
    ax_a.text(370, overall_min, f'Min CF: {overall_min:.1%}', va='center', fontsize=6,
              color=COLORS['secondary'])

    ax_a.set_xlim(1, 365)
    ax_a.set_ylim(0, 0.8)
    ax_a.set_xlabel('Day of Year')
    ax_a.set_ylabel('Capacity Factor')
    ax_a.set_title('a) European Wind Capacity Factor (2015-2024, normalized by installed capacity)',
                   loc='left', fontweight='bold')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Add colorbar for years
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(2015, 2024))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_a, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Year', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # =========================================================================
    # Panel B: 10-Year Baseload Evolution
    # =========================================================================
    years_b = baseload_df['year'].values
    mins = baseload_df['min'].values / 1000  # Convert to GW
    means = baseload_df['mean'].values / 1000

    x = np.arange(len(years_b))
    width = 0.35

    bars_min = ax_b.bar(x - width/2, mins, width, label='Minimum', color=COLORS['secondary'], alpha=0.8)
    bars_mean = ax_b.bar(x + width/2, means, width, label='Mean', color=COLORS['primary'], alpha=0.8)

    # Add trend line for minimum
    z = np.polyfit(x, mins, 1)
    p = np.poly1d(z)
    ax_b.plot(x, p(x), '--', color=COLORS['highlight'], linewidth=1.5, label='Min trend')

    # Calculate growth rate
    growth_per_year = z[0]
    ax_b.text(0.95, 0.95, f'+{growth_per_year:.2f} GW/year',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=7,
              color=COLORS['highlight'], fontweight='bold')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels([str(y) for y in years_b], rotation=45, ha='right')
    ax_b.set_ylabel('Wind Power [GW]')
    ax_b.set_title('b) 10-Year Evolution', loc='left', fontweight='bold')
    ax_b.legend(loc='upper left', frameon=False, fontsize=6)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # =========================================================================
    # Panel C: Capacity Factor Distribution Histogram
    # =========================================================================
    # Combine all years for histogram (use capacity factor)
    all_cf = []
    for year in range(2015, 2025):
        ts = get_aggregated_timeseries(country_data, year)
        if ts is not None:
            capacity_gw = INSTALLED_CAPACITY[year]
            cf = (ts / 1000) / capacity_gw
            all_cf.extend(cf.values)

    all_cf = np.array(all_cf)
    all_cf = all_cf[all_cf > 0]  # Remove zeros for Weibull fit

    # Create histogram
    n, bins, patches = ax_c.hist(all_cf, bins=50, density=True, alpha=0.7,
                                  color=COLORS['fill'], edgecolor=COLORS['primary'],
                                  linewidth=0.5)

    # Mark the "floor" - the minimum never goes to zero
    floor_val = np.min(all_cf)
    ax_c.axvline(floor_val, color=COLORS['secondary'], linestyle='-', linewidth=2)

    # Add annotation for floor
    ax_c.annotate(f'Floor\n{floor_val:.1%}',
                  xy=(floor_val, 0), xytext=(floor_val + 0.1, 3),
                  fontsize=6, ha='left',
                  arrowprops=dict(arrowstyle='->', color=COLORS['secondary']))

    ax_c.set_xlabel('Capacity Factor')
    ax_c.set_ylabel('Density')
    ax_c.set_title('c) CF Distribution', loc='left', fontweight='bold')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.set_xlim(0, 0.7)

    # Add statistics box
    stats_text = f'n = {len(all_cf):,} hours\nMean CF = {np.mean(all_cf):.1%}'
    ax_c.text(0.95, 0.75, stats_text, transform=ax_c.transAxes,
              ha='right', va='top', fontsize=6,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLORS['neutral'], alpha=0.8))

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'figure1_baseload.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")

    return output_path


def main():
    print("=" * 60)
    print("CREATING FIGURE 1: BASELOAD ANALYSIS")
    print("=" * 60)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    print("\nLoading baseload statistics...")
    baseload_df = pd.read_csv(os.path.join(RESULTS_DIR, 'baseload_by_year.csv'))
    print(baseload_df[['year', 'min', 'mean']].to_string(index=False))

    print("\nCreating figure...")
    output_path = create_figure1(country_data, baseload_df)

    print("\n" + "=" * 60)
    print("FIGURE 1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"Finished at {datetime.now()}")
