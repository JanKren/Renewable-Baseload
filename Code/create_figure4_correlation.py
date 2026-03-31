# -*- coding: utf-8 -*-
"""
Figure 4: Correlation Structure

Publication-quality figure for Nature Communications showing:
- Panel A: Geographic-ordered correlation heatmap
- Panel B: Correlation vs distance (conceptual)
- Panel C: Correlation statistics summary

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nature Communications style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
})

# Geographic ordering of countries (roughly West to East, North to South)
GEOGRAPHIC_ORDER = [
    'PT', 'ES', 'IE', 'GB', 'FR', 'BE', 'NL', 'LU', 'DE', 'DK',
    'NO', 'SE', 'FI', 'EE', 'LV', 'LT', 'PL', 'CZ', 'SK', 'AT',
    'CH', 'SI', 'HR', 'HU', 'RO', 'BG', 'RS', 'GR', 'IT'
]

# Approximate country coordinates (lat, lon) for distance calculation
COUNTRY_COORDS = {
    'PT': (39.4, -8.2), 'ES': (40.5, -3.7), 'IE': (53.4, -8.2),
    'GB': (52.4, -1.5), 'FR': (46.2, 2.2), 'BE': (50.5, 4.5),
    'NL': (52.1, 5.3), 'LU': (49.8, 6.1), 'DE': (51.2, 10.5),
    'DK': (56.3, 9.5), 'NO': (60.5, 8.5), 'SE': (60.1, 18.6),
    'FI': (61.9, 25.7), 'EE': (58.6, 25.0), 'LV': (56.9, 24.6),
    'LT': (55.2, 23.9), 'PL': (51.9, 19.1), 'CZ': (49.8, 15.5),
    'SK': (48.7, 19.7), 'AT': (47.5, 14.6), 'CH': (46.8, 8.2),
    'SI': (46.2, 14.9), 'HR': (45.1, 15.2), 'HU': (47.2, 19.5),
    'RO': (45.9, 25.0), 'BG': (42.7, 25.5), 'RS': (44.0, 21.0),
    'GR': (39.1, 21.8), 'IT': (41.9, 12.6)
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


def compute_correlation_matrix(country_data, year=2024):
    """Compute correlation matrix for a specific year."""
    year_data = {}

    for country, df in country_data.items():
        year_df = df[df.index.year == year]
        if len(year_df) > 1000:
            hourly = year_df['wind'].resample('h').mean().dropna()
            if len(hourly) > 1000:
                year_data[country] = hourly

    # Find common timestamps
    common_index = None
    for country, series in year_data.items():
        if common_index is None:
            common_index = set(series.index)
        else:
            common_index = common_index.intersection(set(series.index))

    common_index = sorted(list(common_index))

    if len(common_index) < 100:
        return None, []

    # Create aligned DataFrame
    aligned_df = pd.DataFrame({c: year_data[c].loc[common_index].values
                               for c in year_data.keys()})

    # Compute correlation matrix
    corr_matrix = aligned_df.corr()

    return corr_matrix, list(year_data.keys())


def haversine_distance(coord1, coord2):
    """Calculate distance between two points in km."""
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371 * c  # Earth radius in km


def create_figure4(country_data):
    """Create Figure 4: Correlation Structure multi-panel figure."""

    # Compute correlation matrix
    corr_matrix, countries = compute_correlation_matrix(country_data, 2024)

    if corr_matrix is None:
        print("Error: Could not compute correlation matrix")
        return None

    # Reorder by geographic position
    ordered_countries = [c for c in GEOGRAPHIC_ORDER if c in countries]

    # Add any countries not in the geographic order
    for c in countries:
        if c not in ordered_countries:
            ordered_countries.append(c)

    corr_ordered = corr_matrix.loc[ordered_countries, ordered_countries]

    # Create figure
    fig = plt.figure(figsize=(7.08, 4.72))  # 180mm x 120mm

    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1],
                          wspace=0.25,
                          left=0.08, right=0.96, top=0.90, bottom=0.12)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # =========================================================================
    # Panel A: Correlation Heatmap
    # =========================================================================
    mask = np.triu(np.ones_like(corr_ordered, dtype=bool), k=1)

    sns.heatmap(corr_ordered, ax=ax_a, mask=mask,
                cmap='RdBu_r', center=0, vmin=-0.2, vmax=1,
                square=True, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.6},
                annot=False)

    ax_a.set_title('a) Wind Correlation Matrix (2024)', loc='left', fontweight='bold')
    ax_a.set_xlabel('')
    ax_a.set_ylabel('')

    # Rotate labels
    ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=45, ha='right')
    ax_a.set_yticklabels(ax_a.get_yticklabels(), rotation=0)

    # =========================================================================
    # Panel B: Correlation vs Distance
    # =========================================================================
    # Compute distance and correlation pairs
    pairs = []
    for i, c1 in enumerate(ordered_countries):
        for j, c2 in enumerate(ordered_countries):
            if i < j and c1 in COUNTRY_COORDS and c2 in COUNTRY_COORDS:
                dist = haversine_distance(COUNTRY_COORDS[c1], COUNTRY_COORDS[c2])
                corr = corr_ordered.loc[c1, c2]
                pairs.append({'c1': c1, 'c2': c2, 'distance': dist, 'correlation': corr})

    pairs_df = pd.DataFrame(pairs)

    # Scatter plot
    ax_b.scatter(pairs_df['distance'], pairs_df['correlation'],
                 alpha=0.5, s=20, c='#2166ac', edgecolors='white', linewidths=0.3)

    # Fit trend line
    z = np.polyfit(pairs_df['distance'], pairs_df['correlation'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, pairs_df['distance'].max(), 100)
    ax_b.plot(x_line, p(x_line), '--', color='#d6604d', linewidth=1.5,
              label=f'Trend: r = {z[0]*1000:.2f}/1000km')

    # Add reference lines
    ax_b.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax_b.text(50, 0.52, 'Moderate', fontsize=6, color='gray')
    ax_b.axhline(0.3, color='gray', linestyle=':', alpha=0.5)
    ax_b.text(50, 0.32, 'Weak', fontsize=6, color='gray')

    ax_b.set_xlabel('Distance [km]')
    ax_b.set_ylabel('Correlation')
    ax_b.set_title('b) Correlation vs Distance', loc='left', fontweight='bold')
    ax_b.set_xlim(0, pairs_df['distance'].max() * 1.05)
    ax_b.set_ylim(-0.1, 1.0)
    ax_b.legend(loc='upper right', frameon=False, fontsize=6)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Add statistics box
    stats_text = (f"n = {len(pairs_df)} pairs\n"
                  f"Mean r = {pairs_df['correlation'].mean():.2f}\n"
                  f"Adjacent (< 500 km): {pairs_df[pairs_df['distance'] < 500]['correlation'].mean():.2f}\n"
                  f"Distant (> 1500 km): {pairs_df[pairs_df['distance'] > 1500]['correlation'].mean():.2f}")
    ax_b.text(0.95, 0.05, stats_text, transform=ax_b.transAxes,
              ha='right', va='bottom', fontsize=6,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='gray', alpha=0.9))

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'figure4_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")

    return output_path


def main():
    print("=" * 60)
    print("CREATING FIGURE 4: CORRELATION STRUCTURE")
    print("=" * 60)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    print("\nCreating figure...")
    output_path = create_figure4(country_data)

    print("\n" + "=" * 60)
    print("FIGURE 4 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"Finished at {datetime.now()}")
