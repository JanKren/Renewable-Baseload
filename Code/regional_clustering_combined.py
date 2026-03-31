#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regional Clustering Analysis - Combined Wind + Solar

Extends regional analysis to include combined wind+solar portfolios,
comparing the interconnection dividend for wind-only vs combined.

Authors: Kren & Zajec
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# Configuration
WIND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
SOLAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Solar")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Regional")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define regional clusters
REGIONAL_CLUSTERS = {
    'Northwest Europe': ['DE', 'NL', 'BE', 'FR', 'LU', 'GB', 'IE'],
    'Nordic': ['NO', 'SE', 'DK', 'FI'],
    'Iberia': ['ES', 'PT'],
    'Central Europe': ['AT', 'CH', 'CZ', 'SK', 'HU', 'PL', 'SI'],
    'Southeast Europe': ['RO', 'BG', 'GR', 'RS', 'HR'],
    'Baltics': ['EE', 'LV', 'LT'],
    'Italy': ['IT'],
}

# Colors for regions
REGION_COLORS = {
    'Northwest Europe': '#1f77b4',
    'Nordic': '#2ca02c',
    'Iberia': '#d62728',
    'Central Europe': '#9467bd',
    'Southeast Europe': '#8c564b',
    'Baltics': '#e377c2',
    'Italy': '#ff7f0e',
    'Pan-European': '#333333',
}

# Plot style
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


def load_wind_data(data_dir):
    """Load all country wind production data."""
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
            df.index = df.index.tz_localize(None)

            if 'Wind Total' in df.columns:
                wind = df['Wind Total']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                wind = df[numeric_cols].fillna(0).sum(axis=1)

            wind_hourly = wind.resample('h').mean()
            country_data[country_code] = wind_hourly

        except Exception as e:
            print(f"  Error loading wind {country_code}: {e}")

    return country_data


def load_solar_data(data_dir):
    """Load all country solar production data."""
    country_data = {}
    files = glob.glob(os.path.join(data_dir, "*_solar_2015_2024.csv"))

    for filepath in files:
        filename = os.path.basename(filepath)
        country_code = filename.split('_')[0]

        try:
            df = pd.read_csv(filepath)
            date_col = df.columns[0]
            df['datetime'] = pd.to_datetime(df[date_col], utc=True)
            df = df.set_index('datetime')
            df.index = df.index.tz_localize(None)

            if 'Solar' in df.columns:
                solar = df['Solar']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                solar = df[numeric_cols].fillna(0).sum(axis=1)

            solar_hourly = solar.resample('h').mean()
            country_data[country_code] = solar_hourly

        except Exception as e:
            print(f"  Error loading solar {country_code}: {e}")

    return country_data


def compute_regional_aggregates(wind_data, solar_data, year=None, combined=False):
    """Compute regional and pan-European aggregates."""

    # Find common countries
    if combined:
        common_countries = set(wind_data.keys()) & set(solar_data.keys())
    else:
        common_countries = set(wind_data.keys())

    # Filter by year if specified
    if year:
        wind_filtered = {k: v[v.index.year == year] for k, v in wind_data.items() if k in common_countries}
        if combined:
            solar_filtered = {k: v[v.index.year == year] for k, v in solar_data.items() if k in common_countries}
    else:
        wind_filtered = {k: v for k, v in wind_data.items() if k in common_countries}
        if combined:
            solar_filtered = {k: v for k, v in solar_data.items() if k in common_countries}

    # Find common timestamps - only from wind data (solar will be reindexed)
    all_indices = [set(v.index) for v in wind_filtered.values() if len(v) > 0]

    common_idx = set.intersection(*all_indices) if all_indices else set()
    common_idx = sorted(common_idx)

    if len(common_idx) < 1000:
        print(f"  Warning: Only {len(common_idx)} common timestamps")
        return None, None

    # Align all countries
    aligned_wind = {}
    aligned_solar = {}
    for country in common_countries:
        if country in wind_filtered:
            aligned_wind[country] = wind_filtered[country].reindex(common_idx).fillna(0).values
        if combined and country in solar_filtered:
            aligned_solar[country] = solar_filtered[country].reindex(common_idx).fillna(0).values

    # Compute regional aggregates
    regional_totals = {}
    for region, countries in REGIONAL_CLUSTERS.items():
        available = [c for c in countries if c in aligned_wind]
        if available:
            wind_total = np.sum([aligned_wind[c] for c in available], axis=0)
            if combined:
                solar_available = [c for c in available if c in aligned_solar]
                solar_total = np.sum([aligned_solar[c] for c in solar_available], axis=0) if solar_available else 0
                total = wind_total + solar_total
            else:
                total = wind_total
            regional_totals[region] = pd.Series(total, index=common_idx)

    # Pan-European total
    wind_pan = np.sum([aligned_wind[c] for c in aligned_wind.keys()], axis=0)
    if combined:
        solar_pan = np.sum([aligned_solar[c] for c in aligned_solar.keys()], axis=0)
        pan_european = wind_pan + solar_pan
    else:
        pan_european = wind_pan
    regional_totals['Pan-European'] = pd.Series(pan_european, index=common_idx)

    return regional_totals, len(common_countries)


def compute_statistics(regional_totals):
    """Compute statistics for each region."""
    stats = []

    for region, series in regional_totals.items():
        stats.append({
            'Region': region,
            'Min [GW]': series.min() / 1000,
            'Mean [GW]': series.mean() / 1000,
            'Max [GW]': series.max() / 1000,
            'CV': series.std() / series.mean(),
        })

    return pd.DataFrame(stats)


def create_combined_regional_figure(wind_stats, combined_stats, wind_totals, combined_totals, output_dir):
    """Create regional clustering figure comparing wind-only vs combined."""

    fig = plt.figure(figsize=(7.08, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1],
                          hspace=0.45, wspace=0.25,
                          left=0.10, right=0.95, top=0.94, bottom=0.08)

    # Panel A uses a split axis: regional bars (left) and Pan-European (right)
    # with independent y-scales so that the small regional bars are readable.
    gs_a = gs[0, 0].subgridspec(1, 2, width_ratios=[5, 1.4], wspace=0.08)
    ax_a_reg = fig.add_subplot(gs_a[0])   # regional bars
    ax_a_pan = fig.add_subplot(gs_a[1])   # Pan-European bars

    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    all_regions = list(REGIONAL_CLUSTERS.keys()) + ['Pan-European']
    reg_regions = list(REGIONAL_CLUSTERS.keys())
    x_reg = np.arange(len(reg_regions))
    width = 0.35

    # Panel A: Baseload floor comparison (wind vs combined) — split axes
    wind_mins_all = [wind_stats[wind_stats['Region'] == r]['Min [GW]'].values[0] for r in all_regions]
    combined_mins_all = [combined_stats[combined_stats['Region'] == r]['Min [GW]'].values[0] for r in all_regions]

    wind_mins_reg = wind_mins_all[:-1]
    combined_mins_reg = combined_mins_all[:-1]
    wind_min_pan = wind_mins_all[-1]
    combined_min_pan = combined_mins_all[-1]

    # Left: regional bars
    bars1 = ax_a_reg.bar(x_reg - width/2, wind_mins_reg, width, label='Wind only',
                          color='#2166ac', edgecolor='white')
    bars2 = ax_a_reg.bar(x_reg + width/2, combined_mins_reg, width, label='Wind + Solar',
                          color='#b2182b', edgecolor='white')

    for bar, val in zip(bars1, wind_mins_reg):
        if val > 0.05:
            ax_a_reg.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                          f'{val:.1f}', ha='center', va='bottom', fontsize=5)
    for bar, val in zip(bars2, combined_mins_reg):
        if val > 0.05:
            ax_a_reg.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                          f'{val:.1f}', ha='center', va='bottom', fontsize=5)

    ax_a_reg.set_xticks(x_reg)
    ax_a_reg.set_xticklabels(reg_regions, rotation=45, ha='right', fontsize=6)
    ax_a_reg.set_ylabel('Minimum Production (GW)')
    ax_a_reg.set_title('a) Baseload floor by region', loc='left', fontweight='bold')
    ax_a_reg.legend(loc='upper center', frameon=False, fontsize=6)
    ax_a_reg.spines['top'].set_visible(False)
    ax_a_reg.spines['right'].set_visible(False)

    # Right: Pan-European bars (separate y-scale)
    x_pan = np.array([0])
    ax_a_pan.bar(x_pan - width/2, [wind_min_pan], width, color='#2166ac', edgecolor='white')
    ax_a_pan.bar(x_pan + width/2, [combined_min_pan], width, color='#b2182b', edgecolor='white')

    ax_a_pan.text(-width/2, wind_min_pan + 0.3, f'{wind_min_pan:.1f}',
                   ha='center', va='bottom', fontsize=5)
    ax_a_pan.text(width/2, combined_min_pan + 0.3, f'{combined_min_pan:.1f}',
                   ha='center', va='bottom', fontsize=5)

    ax_a_pan.set_xticks(x_pan)
    ax_a_pan.set_xticklabels(['Pan-\nEuropean'], fontsize=6)
    ax_a_pan.yaxis.set_visible(False)
    ax_a_pan.spines['top'].set_visible(False)
    ax_a_pan.spines['left'].set_visible(False)
    ax_a_pan.spines['right'].set_visible(False)

    # Keep regions list for panels B, C, D
    regions = all_regions
    x = np.arange(len(regions))

    # Panel B: CV comparison
    wind_cvs = [wind_stats[wind_stats['Region'] == r]['CV'].values[0] for r in regions]
    combined_cvs = [combined_stats[combined_stats['Region'] == r]['CV'].values[0] for r in regions]

    ax_b.bar(x - width/2, wind_cvs, width, label='Wind only', color='#2166ac', edgecolor='white')
    ax_b.bar(x + width/2, combined_cvs, width, label='Wind + Solar', color='#b2182b', edgecolor='white')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(regions, rotation=45, ha='right', fontsize=6)
    ax_b.set_ylabel('CV', fontsize=7)
    ax_b.set_title('b) Coefficient of variation', loc='left', fontweight='bold')
    ax_b.legend(loc='upper right', frameon=False, fontsize=6)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Panel C: Time series comparison
    start_date = '2024-06-15'
    end_date = '2024-06-22'

    # Plot Pan-European first so it appears on top in legend
    for region in ['Pan-European', 'Northwest Europe']:
        if region in wind_totals:
            wind_series = wind_totals[region] / 1000
            combined_series = combined_totals[region] / 1000
            wind_subset = wind_series[start_date:end_date]
            combined_subset = combined_series[start_date:end_date]

            lw = 1.5 if region == 'Pan-European' else 0.8
            alpha = 1.0 if region == 'Pan-European' else 0.5
            short = 'Pan-EU' if region == 'Pan-European' else 'NW-EU'
            ax_c.plot(range(len(wind_subset)), wind_subset.values,
                     color='#2166ac', linestyle='--', alpha=alpha, linewidth=lw,
                     label=f'{short} (Wind)')
            ax_c.plot(range(len(combined_subset)), combined_subset.values,
                     color='#b2182b', alpha=alpha, linewidth=lw,
                     label=f'{short} (W+S)')

    ax_c.set_xlabel('Hour (June 15-22, 2024)')
    ax_c.set_ylabel('Production (GW)')
    ax_c.set_title('c) Regional vs pan-European', loc='left', fontweight='bold')
    # Reorder legend: Pan-EU first, then NW-EU
    handles, labels = ax_c.get_legend_handles_labels()
    # Current order: Pan-EU (Wind), Pan-EU (W+S), NW-EU (Wind), NW-EU (W+S)
    # Already correct, but ensure it stays this way
    ax_c.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=6)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Panel D: Interconnection dividend comparison
    wind_regional_sum = sum([wind_stats[wind_stats['Region'] == r]['Min [GW]'].values[0]
                            for r in REGIONAL_CLUSTERS.keys()])
    wind_pan_eu = wind_stats[wind_stats['Region'] == 'Pan-European']['Min [GW]'].values[0]
    wind_dividend = wind_pan_eu / wind_regional_sum

    combined_regional_sum = sum([combined_stats[combined_stats['Region'] == r]['Min [GW]'].values[0]
                                 for r in REGIONAL_CLUSTERS.keys()])
    combined_pan_eu = combined_stats[combined_stats['Region'] == 'Pan-European']['Min [GW]'].values[0]
    combined_dividend = combined_pan_eu / combined_regional_sum

    bar_positions = [0, 1, 2.5, 3.5]
    bar_values = [wind_regional_sum, wind_pan_eu, combined_regional_sum, combined_pan_eu]
    bar_colors = ['#a6cee3', '#2166ac', '#fbb4ae', '#b2182b']

    bars = ax_d.bar(bar_positions, bar_values, color=bar_colors, edgecolor='black', linewidth=0.5)

    ax_d.set_xticks([0.5, 3])
    ax_d.set_xticklabels(['Wind only', 'Wind + Solar'], fontsize=8)
    ax_d.set_ylabel('Baseload Floor (GW)')
    ax_d.set_title('d) Interconnection dividend', loc='left', fontweight='bold')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # Add dividend annotations
    ax_d.annotate('', xy=(1, wind_pan_eu), xytext=(0, wind_regional_sum),
                  arrowprops=dict(arrowstyle='->', color='#2166ac', lw=2))
    ax_d.text(0.5, (wind_regional_sum + wind_pan_eu) / 2 + 0.5,
              f'{wind_dividend:.1f}x', ha='center', fontsize=8, fontweight='bold', color='#2166ac')

    ax_d.annotate('', xy=(3.5, combined_pan_eu), xytext=(2.5, combined_regional_sum),
                  arrowprops=dict(arrowstyle='->', color='#b2182b', lw=2))
    ax_d.text(2.5, (combined_regional_sum + combined_pan_eu) / 2 + 0.5,
              f'{combined_dividend:.1f}x', ha='center', fontsize=8, fontweight='bold', color='#b2182b')

    # Value labels
    for bar, val in zip(bars, bar_values):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                  f'{val:.1f}', ha='center', va='bottom', fontsize=6)

    # Legend for panel D
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2166ac', edgecolor='black', label='Pan-European'),
                       Patch(facecolor='#a6cee3', edgecolor='black', label='Sum of regional')]
    ax_d.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=6)

    # Save
    output_path = os.path.join(output_dir, 'figure_regional_clustering.pdf')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")

    return wind_dividend, combined_dividend


def main():
    print("=" * 70)
    print("REGIONAL CLUSTERING ANALYSIS - WIND + SOLAR")
    print("=" * 70)

    print("\nLoading wind data...")
    wind_data = load_wind_data(WIND_DIR)
    print(f"  Loaded {len(wind_data)} countries")

    print("\nLoading solar data...")
    solar_data = load_solar_data(SOLAR_DIR)
    print(f"  Loaded {len(solar_data)} countries")

    common = set(wind_data.keys()) & set(solar_data.keys())
    print(f"\nCountries with both wind and solar: {len(common)}")

    print("\n" + "=" * 70)
    print("WIND-ONLY ANALYSIS (2024)")
    print("=" * 70)

    wind_totals, n_wind = compute_regional_aggregates(wind_data, solar_data, year=2024, combined=False)
    if wind_totals is None:
        print("Error computing wind aggregates")
        return

    wind_stats = compute_statistics(wind_totals)
    print(wind_stats.to_string(index=False))

    print("\n" + "=" * 70)
    print("COMBINED WIND+SOLAR ANALYSIS (2024)")
    print("=" * 70)

    combined_totals, n_combined = compute_regional_aggregates(wind_data, solar_data, year=2024, combined=True)
    if combined_totals is None:
        print("Error computing combined aggregates")
        return

    combined_stats = compute_statistics(combined_totals)
    print(combined_stats.to_string(index=False))

    print("\n" + "=" * 70)
    print("COMPARISON: INTERCONNECTION DIVIDEND")
    print("=" * 70)

    # Wind-only
    wind_regional_sum = sum([wind_stats[wind_stats['Region'] == r]['Min [GW]'].values[0]
                            for r in REGIONAL_CLUSTERS.keys()])
    wind_pan_eu = wind_stats[wind_stats['Region'] == 'Pan-European']['Min [GW]'].values[0]

    # Combined
    combined_regional_sum = sum([combined_stats[combined_stats['Region'] == r]['Min [GW]'].values[0]
                                 for r in REGIONAL_CLUSTERS.keys()])
    combined_pan_eu = combined_stats[combined_stats['Region'] == 'Pan-European']['Min [GW]'].values[0]

    print(f"\nWIND ONLY:")
    print(f"  Sum of regional minimums: {wind_regional_sum:.2f} GW")
    print(f"  Pan-European minimum:     {wind_pan_eu:.2f} GW")
    print(f"  Interconnection dividend: {wind_pan_eu/wind_regional_sum:.2f}x")

    print(f"\nCOMBINED WIND + SOLAR:")
    print(f"  Sum of regional minimums: {combined_regional_sum:.2f} GW")
    print(f"  Pan-European minimum:     {combined_pan_eu:.2f} GW")
    print(f"  Interconnection dividend: {combined_pan_eu/combined_regional_sum:.2f}x")

    print(f"\nIMPROVEMENT FROM ADDING SOLAR:")
    print(f"  Regional sum: +{combined_regional_sum - wind_regional_sum:.2f} GW ({(combined_regional_sum/wind_regional_sum - 1)*100:.0f}%)")
    print(f"  Pan-European: +{combined_pan_eu - wind_pan_eu:.2f} GW ({(combined_pan_eu/wind_pan_eu - 1)*100:.0f}%)")

    print("\nCreating figure...")
    wind_div, combined_div = create_combined_regional_figure(
        wind_stats, combined_stats, wind_totals, combined_totals, OUTPUT_DIR)

    # Save results
    results = {
        'wind_regional_sum': wind_regional_sum,
        'wind_pan_eu': wind_pan_eu,
        'wind_dividend': wind_pan_eu / wind_regional_sum,
        'combined_regional_sum': combined_regional_sum,
        'combined_pan_eu': combined_pan_eu,
        'combined_dividend': combined_pan_eu / combined_regional_sum,
    }

    pd.DataFrame([results]).to_csv(
        os.path.join(RESULTS_DIR, 'regional_combined_comparison.csv'), index=False)

    wind_stats.to_csv(os.path.join(RESULTS_DIR, 'regional_statistics_wind.csv'), index=False)
    combined_stats.to_csv(os.path.join(RESULTS_DIR, 'regional_statistics_combined.csv'), index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
