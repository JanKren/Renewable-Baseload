# -*- coding: utf-8 -*-
"""
Create Figure 3: European Wind Correlation Map

Publication-quality map for Nature Communications showing:
  a) Regional clusters with correlation network edges
  b) Marginal diversification value by country

Authors: Kren, Zajec & Tiselj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import glob

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Data', 'Wind')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'Results')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'Figures_Paper')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Country centroids (lon, lat) — adjusted to reduce overlap
# =============================================================================
COUNTRY_COORDS = {
    'PT': (-8.5, 39.4),
    'ES': (-3.7, 40.2),
    'IE': (-8.5, 53.5),
    'GB': (-2.0, 54.5),
    'FR': (2.2, 46.6),
    'BE': (4.0, 50.8),
    'NL': (5.5, 52.5),
    'LU': (6.1, 49.6),
    'DE': (10.4, 51.2),
    'DK': (9.5, 56.3),
    'NO': (8.0, 62.0),
    'SE': (16.0, 62.5),
    'FI': (26.0, 64.0),
    'EE': (25.5, 59.0),
    'LV': (24.5, 57.0),
    'LT': (23.8, 55.2),
    'PL': (19.5, 52.0),
    'CZ': (15.5, 49.8),
    'SK': (19.5, 48.7),
    'AT': (14.0, 47.3),
    'CH': (8.0, 46.8),
    'SI': (14.8, 46.0),
    'HR': (16.2, 44.8),
    'HU': (19.5, 47.2),
    'RO': (25.5, 46.0),
    'BG': (25.5, 42.8),
    'RS': (21.0, 44.0),
    'GR': (22.5, 39.0),
    'IT': (12.0, 42.5),
}

# Label offsets (dx, dy in degrees) to avoid node overlap
LABEL_OFFSETS = {
    'PT': (-3.0, -1.0),
    'ES': (0, -2.0),
    'IE': (-2.5, 0),
    'GB': (-3.0, 0),
    'FR': (-2.5, -1.0),
    'BE': (-2.5, 0.5),
    'NL': (-2.5, 1.0),
    'LU': (-2.8, -0.5),
    'DE': (2.5, 0),
    'DK': (-2.5, 0.5),
    'NO': (-3.0, 0),
    'SE': (2.5, 0),
    'FI': (2.5, 0),
    'EE': (3.0, 0),
    'LV': (3.0, 0),
    'LT': (3.0, 0),
    'PL': (2.5, 0),
    'CZ': (2.5, 0),
    'SK': (2.5, 0),
    'AT': (-2.8, 0),
    'CH': (-2.5, 0),
    'SI': (-2.5, -1.0),
    'HR': (-2.5, -0.8),
    'HU': (2.5, 0),
    'RO': (2.5, 0),
    'BG': (2.5, 0),
    'RS': (-2.5, 0),
    'GR': (2.5, 0),
    'IT': (-2.5, 0),
}

# =============================================================================
# Regional clusters
# =============================================================================
CLUSTERS = {
    'Northwestern': ['DE', 'NL', 'BE', 'FR', 'LU', 'GB', 'IE'],
    'Nordic':       ['NO', 'SE', 'DK', 'FI'],
    'Iberia':       ['ES', 'PT'],
    'Central':      ['AT', 'CH', 'CZ', 'SK', 'HU', 'PL', 'SI'],
    'Southeastern': ['RO', 'BG', 'GR', 'RS', 'HR'],
    'Baltic':       ['EE', 'LV', 'LT'],
    'Italy':        ['IT'],
}

CLUSTER_COLORS = {
    'Northwestern': '#2166ac',
    'Nordic':       '#1b9e77',
    'Iberia':       '#d95f02',
    'Central':      '#e7298a',
    'Southeastern': '#e31a1c',
    'Baltic':       '#7570b3',
    'Italy':        '#66a61e',
}

COUNTRY_CLUSTER = {}
for cluster, countries in CLUSTERS.items():
    for c in countries:
        COUNTRY_CLUSTER[c] = cluster

RENAME = {'UK_gridwatch': 'GB'}

# =============================================================================
# Style
# =============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
})


def compute_correlation_matrix():
    """Compute correlation matrix from raw wind data (all 29 countries)."""
    files = glob.glob(os.path.join(DATA_DIR, '*_wind_2015_2024.csv'))
    all_series = {}
    for filepath in sorted(files):
        code = os.path.basename(filepath).split('_')[0]
        code = RENAME.get(code, code)
        if code not in COUNTRY_COORDS:
            continue
        df = pd.read_csv(filepath)
        date_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[date_col], utc=True)
        df = df.set_index('datetime')
        df = df.drop(columns=[date_col], errors='ignore')
        if 'Wind Total' in df.columns:
            series = df['Wind Total']
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            series = df[numeric_cols].fillna(0).sum(axis=1)
        all_series[code] = series.resample('h').mean()

    combined = pd.DataFrame(all_series).dropna()
    corr = combined.corr()
    print(f"  Correlation matrix: {len(corr)} countries")
    return corr


def load_data():
    """Load correlation matrix and marginal diversification."""
    corr = compute_correlation_matrix()

    # Marginal diversification
    path = os.path.join(RESULTS_DIR, 'Diversification', 'marginal_diversification.csv')
    marginal = pd.read_csv(path)

    return corr, marginal


def get_mean_production(marginal):
    """Extract per-country mean production from the greedy build-up."""
    mean_prod = {}
    prev = 0
    for _, row in marginal.iterrows():
        c = RENAME.get(row['country_added'], row['country_added'])
        mean_prod[c] = row['portfolio_mean_gw'] - prev
        prev = row['portfolio_mean_gw']
    return mean_prod


def draw_panel_a(ax, corr, mean_prod):
    """Panel a: Correlation network with regional clusters."""

    countries = [c for c in corr.columns if c in COUNTRY_COORDS]

    # Background map
    ax.set_extent([-13, 33, 34, 69], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f7f7f7', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f0f5')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#d0d0d0')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#aaaaaa')

    # Collect edges — different thresholds for positive/negative
    edges = []
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if j <= i:
                continue
            r = corr.loc[c1, c2]
            if np.isnan(r):
                continue
            if r > 0 and r < 0.25:
                continue
            if r < 0 and r > -0.08:
                continue
            edges.append((c1, c2, r))

    # Sort: weak edges first, strong on top
    edges.sort(key=lambda x: abs(x[2]))

    for c1, c2, r in edges:
        lon1, lat1 = COUNTRY_COORDS[c1]
        lon2, lat2 = COUNTRY_COORDS[c2]
        if r > 0:
            alpha = 0.08 + 0.45 * ((r - 0.25) / 0.75)
            width = 0.3 + 1.8 * r
            color = (0.13, 0.40, 0.67, alpha)  # blue
        else:
            alpha = 0.25 + 0.5 * min(1.0, (abs(r) - 0.08) / 0.15)
            width = 0.5 + 3.0 * abs(r)
            color = (0.89, 0.10, 0.11, alpha)  # red
        ax.plot([lon1, lon2], [lat1, lat2], '-', color=color,
                linewidth=width, transform=ccrs.PlateCarree(), zorder=2)

    # Draw nodes
    for c in countries:
        lon, lat = COUNTRY_COORDS[c]
        cluster = COUNTRY_CLUSTER.get(c, 'Central')
        color = CLUSTER_COLORS[cluster]
        prod = mean_prod.get(c, 1.0)
        size = max(50, min(280, 40 + 15 * prod))

        ax.scatter(lon, lat, s=size, c=color, edgecolors='white',
                   linewidths=0.7, zorder=5, transform=ccrs.PlateCarree())

    # Labels outside nodes
    for c in countries:
        lon, lat = COUNTRY_COORDS[c]
        dx, dy = LABEL_OFFSETS.get(c, (2.0, 0))
        ax.annotate(c, xy=(lon, lat), xytext=(lon + dx, lat + dy),
                    fontsize=5.5, fontweight='bold', color='#333333',
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color='#999999',
                                    lw=0.3, shrinkA=3, shrinkB=0),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    zorder=7)

    # Legend — place upper-left to avoid overlapping Iberia
    handles = [mpatches.Patch(facecolor=CLUSTER_COLORS[cl], edgecolor='white',
                              label=cl, linewidth=0.5)
               for cl in CLUSTERS]
    # Add edge legend
    handles.append(plt.Line2D([0], [0], color=(0.13, 0.40, 0.67, 0.4),
                              linewidth=1.5, label='r > 0.25'))
    handles.append(plt.Line2D([0], [0], color=(0.89, 0.10, 0.11, 0.5),
                              linewidth=1.5, label='r < \u22120.08'))

    ax.legend(handles=handles, loc='upper left', fontsize=5,
              frameon=True, fancybox=False, edgecolor='#cccccc',
              framealpha=0.92, handlelength=1.2, handleheight=0.8,
              borderpad=0.4, labelspacing=0.25, ncol=1)

    ax.text(0.5, 1.02, 'Correlation network (2024)', fontsize=9,
            transform=ax.transAxes, ha='center', va='bottom')


def draw_panel_b(ax, corr, marginal, mean_prod):
    """Panel b: Marginal diversification value map."""

    countries = [c for c in corr.columns if c in COUNTRY_COORDS]

    # Background
    ax.set_extent([-13, 33, 34, 69], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f7f7f7', edgecolor='none')
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f0f5')
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor='#d0d0d0')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#aaaaaa')

    # Parse marginal data
    marg_values = {}
    marg_rank = {}
    for _, row in marginal.iterrows():
        c = RENAME.get(row['country_added'], row['country_added'])
        marg_values[c] = row['marginal_cv_reduction']
        marg_rank[c] = int(row['step'])

    max_pos = max(v for v in marg_values.values() if v > 0)

    # Use a truncated YlGn colormap that avoids the near-white low end
    from matplotlib.colors import LinearSegmentedColormap
    base_cmap = plt.cm.YlGn
    colors_list = base_cmap(np.linspace(0.15, 1.0, 256))
    cmap = LinearSegmentedColormap.from_list('YlGn_trunc', colors_list)

    # Draw nodes colored by diversification value
    for c in countries:
        if c not in COUNTRY_COORDS or c not in marg_values:
            continue
        lon, lat = COUNTRY_COORDS[c]
        val = marg_values[c]
        rank = marg_rank.get(c, 30)

        if val > 0:
            color = cmap(np.sqrt(val / max_pos))
        else:
            color = '#e0e0e0'  # grey for negative/zero

        prod = mean_prod.get(c, 1.0)
        size = max(50, min(280, 40 + 15 * prod))

        edgecolor = '#222222' if rank <= 5 else '#888888'
        edgewidth = 1.2 if rank <= 5 else 0.5

        ax.scatter(lon, lat, s=size, c=[color], edgecolors=edgecolor,
                   linewidths=edgewidth, zorder=5,
                   transform=ccrs.PlateCarree())

    # Labels: rank for top 10, country code for others
    for c in countries:
        if c not in COUNTRY_COORDS or c not in marg_values:
            continue
        lon, lat = COUNTRY_COORDS[c]
        rank = marg_rank.get(c, 30)
        val = marg_values[c]
        dx, dy = LABEL_OFFSETS.get(c, (2.0, 0))

        if rank <= 10 and val > 0:
            label = f'{c} (#{rank})'
            fw = 'bold'
            fs = 5.5
        else:
            label = c
            fw = 'normal'
            fs = 5
        color_text = '#222222' if rank <= 5 else '#555555'

        ax.annotate(label, xy=(lon, lat), xytext=(lon + dx, lat + dy),
                    fontsize=fs, fontweight=fw, color=color_text,
                    ha='center', va='center',
                    arrowprops=dict(arrowstyle='-', color='#bbbbbb',
                                    lw=0.3, shrinkA=3, shrinkB=0),
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    textcoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    zorder=7)

    # Colorbar using the same truncated colormap
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=max_pos))
    sm.set_array([])

    # Position colorbar inside the panel
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cbar_ax = inset_axes(ax, width="3%", height="30%", loc='lower right',
                         borderpad=1.5)
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('CV reduction [%]', fontsize=5.5, labelpad=2)
    cbar.ax.tick_params(labelsize=5, length=2)

    # Annotation: top 5
    top5_lines = []
    for _, row in marginal.iterrows():
        c = RENAME.get(row['country_added'], row['country_added'])
        val = row['marginal_cv_reduction']
        step = int(row['step'])
        if step == 1:
            top5_lines.append(f"#1  {c:>2s}  (start)")
        elif step <= 6:
            top5_lines.append(f"#{step}  {c:>2s}  \u2212{val:.1f}% CV")
    top5_text = "Diversification ranking:\n" + "\n".join(top5_lines)

    ax.text(0.03, 0.97, top5_text, transform=ax.transAxes,
            fontsize=5.5, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#cccccc', alpha=0.92))

    ax.set_title('b', fontsize=11, fontweight='bold', loc='left', x=-0.02)
    ax.text(0.5, 1.02, 'Marginal diversification value', fontsize=9,
            transform=ax.transAxes, ha='center', va='bottom')


def main():
    print("Creating Figure 3: European correlation map (single panel)...")
    corr, marginal = load_data()
    mean_prod = get_mean_production(marginal)

    proj = ccrs.LambertConformal(central_longitude=10, central_latitude=50)
    fig = plt.figure(figsize=(4.5, 4.5))

    ax1 = fig.add_axes([0.02, 0.02, 0.96, 0.92], projection=proj)

    draw_panel_a(ax1, corr, mean_prod)

    # Save
    for ext in ['png', 'pdf']:
        path = os.path.join(OUTPUT_DIR, f'figure3_map.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {path}")
    plt.close()
    print("Done.")


if __name__ == '__main__':
    main()
