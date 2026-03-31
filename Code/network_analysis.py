# -*- coding: utf-8 -*-
"""
Correlation Network Evolution Analysis

Tracks how the wind production correlation network topology evolved
over 2015-2024 as European wind capacity doubled.

Computes:
- Network modularity (cluster structure)
- Average clustering coefficient
- Network diameter
- Community detection (Louvain algorithm)
- Betweenness centrality (bridge countries)

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from networkx.algorithms import community
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Network")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

YEARS = list(range(2015, 2025))

# Correlation threshold for edge inclusion
EDGE_THRESHOLD = 0.3

# Geographic positions for visualization (approximate)
COUNTRY_POS = {
    'PT': (-1.5, 0), 'ES': (-0.5, 0.5), 'FR': (0.5, 1.5), 'IE': (-1, 2.5),
    'GB': (-0.5, 2.5), 'BE': (0.8, 2), 'NL': (1, 2.3), 'LU': (0.9, 1.8),
    'DE': (1.5, 2), 'DK': (1.5, 2.8), 'NO': (1.5, 3.5), 'SE': (2.2, 3.2),
    'FI': (3, 3.5), 'EE': (3.2, 2.8), 'LV': (3.3, 2.5), 'LT': (3.2, 2.2),
    'PL': (2.5, 2), 'CZ': (2, 1.5), 'SK': (2.5, 1.3), 'AT': (2, 1),
    'CH': (1.2, 1), 'SI': (2.2, 0.8), 'HR': (2.3, 0.5), 'HU': (2.8, 1),
    'RO': (3.5, 0.8), 'BG': (3.5, 0.3), 'RS': (3, 0.3), 'GR': (3.2, -0.5),
    'IT': (1.8, 0)
}

# Regional colors
REGION_COLORS = {
    'Northwestern': '#2166ac',  # Blue
    'Nordic': '#4393c3',        # Light blue
    'Baltic': '#92c5de',        # Very light blue
    'Central': '#f4a582',       # Light orange
    'Southeastern': '#d6604d',  # Red
    'Iberian': '#5aae61',       # Green
}

COUNTRY_REGIONS = {
    'PT': 'Iberian', 'ES': 'Iberian',
    'FR': 'Northwestern', 'GB': 'Northwestern', 'IE': 'Northwestern',
    'BE': 'Northwestern', 'NL': 'Northwestern', 'LU': 'Northwestern', 'DE': 'Northwestern',
    'DK': 'Nordic', 'NO': 'Nordic', 'SE': 'Nordic', 'FI': 'Nordic',
    'EE': 'Baltic', 'LV': 'Baltic', 'LT': 'Baltic', 'PL': 'Baltic',
    'CZ': 'Central', 'SK': 'Central', 'AT': 'Central', 'CH': 'Central',
    'SI': 'Central', 'HU': 'Central',
    'HR': 'Southeastern', 'RO': 'Southeastern', 'BG': 'Southeastern',
    'RS': 'Southeastern', 'GR': 'Southeastern', 'IT': 'Southeastern'
}

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


def compute_correlation_matrix(country_data, year):
    """Compute correlation matrix for a specific year."""
    year_data = {}

    for country, df in country_data.items():
        year_df = df[df.index.year == year]
        if len(year_df) > 1000:
            hourly = year_df['wind'].resample('h').mean().dropna()
            if len(hourly) > 1000:
                year_data[country] = hourly

    if len(year_data) < 5:
        return None, []

    # Find common index
    common_idx = None
    for country, series in year_data.items():
        if common_idx is None:
            common_idx = set(series.index)
        else:
            common_idx = common_idx.intersection(set(series.index))

    common_idx = sorted(list(common_idx))

    if len(common_idx) < 500:
        return None, []

    # Create aligned DataFrame
    countries = sorted(year_data.keys())
    aligned_df = pd.DataFrame({c: year_data[c].loc[common_idx].values
                               for c in countries})

    corr_matrix = aligned_df.corr()

    return corr_matrix, countries


def build_network(corr_matrix, countries, threshold=0.3):
    """Build network graph from correlation matrix."""
    G = nx.Graph()

    # Add nodes
    for c in countries:
        G.add_node(c, region=COUNTRY_REGIONS.get(c, 'Unknown'))

    # Add edges
    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i < j:
                r = corr_matrix.loc[c1, c2]
                if abs(r) >= threshold:
                    G.add_edge(c1, c2, weight=r)

    return G


def compute_network_metrics(G):
    """Compute network topology metrics."""
    if G.number_of_nodes() == 0:
        return {}

    metrics = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
    }

    # Average clustering
    try:
        metrics['avg_clustering'] = nx.average_clustering(G, weight='weight')
    except:
        metrics['avg_clustering'] = np.nan

    # Modularity (using Louvain communities)
    try:
        communities_gen = community.louvain_communities(G, weight='weight', seed=42)
        metrics['n_communities'] = len(communities_gen)
        metrics['modularity'] = community.modularity(G, communities_gen, weight='weight')
    except:
        metrics['n_communities'] = np.nan
        metrics['modularity'] = np.nan

    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight')
        metrics['top_betweenness'] = max(betweenness, key=betweenness.get)
        metrics['max_betweenness'] = betweenness[metrics['top_betweenness']]
    except:
        metrics['top_betweenness'] = None
        metrics['max_betweenness'] = np.nan

    return metrics


def analyze_all_years(country_data, years, threshold=0.3):
    """Analyze network evolution across years."""
    results = []

    for year in years:
        print(f"  {year}...", end=' ')

        corr_matrix, countries = compute_correlation_matrix(country_data, year)

        if corr_matrix is None:
            print("insufficient data")
            continue

        G = build_network(corr_matrix, countries, threshold)
        metrics = compute_network_metrics(G)

        metrics['year'] = year
        metrics['countries'] = len(countries)

        results.append(metrics)
        print(f"nodes={metrics['n_nodes']}, edges={metrics['n_edges']}, mod={metrics.get('modularity', np.nan):.3f}")

    return pd.DataFrame(results)


def create_network_figure(country_data, metrics_df, output_dir):
    """Create network evolution figure."""

    fig = plt.figure(figsize=(7.08, 5.5))

    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.08, right=0.96, top=0.93, bottom=0.08)

    ax_net1 = fig.add_subplot(gs[0, 0])
    ax_net2 = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[1, :])

    # Get networks for 2015 and 2024
    corr_2015, countries_2015 = compute_correlation_matrix(country_data, 2015)
    corr_2024, countries_2024 = compute_correlation_matrix(country_data, 2024)

    G_2015 = build_network(corr_2015, countries_2015, EDGE_THRESHOLD)
    G_2024 = build_network(corr_2024, countries_2024, EDGE_THRESHOLD)

    # Draw 2015 network
    draw_network(G_2015, ax_net1, "a) Network 2015")

    # Draw 2024 network
    draw_network(G_2024, ax_net2, "b) Network 2024")

    # Plot metrics evolution
    ax_metrics.plot(metrics_df['year'], metrics_df['modularity'], 'o-',
                   label='Modularity', color='#2166ac', linewidth=2, markersize=6)
    ax_metrics.plot(metrics_df['year'], metrics_df['avg_clustering'], 's--',
                   label='Avg Clustering', color='#d6604d', linewidth=2, markersize=6)
    ax_metrics.plot(metrics_df['year'], metrics_df['density'], '^:',
                   label='Density', color='#5aae61', linewidth=2, markersize=6)

    ax_metrics.set_xlabel('Year')
    ax_metrics.set_ylabel('Metric Value')
    ax_metrics.set_title('c) Network Topology Evolution', loc='left', fontweight='bold')
    ax_metrics.legend(loc='upper right', frameon=False)
    ax_metrics.spines['top'].set_visible(False)
    ax_metrics.spines['right'].set_visible(False)
    ax_metrics.set_xlim(2014, 2025)

    # Save
    output_path = os.path.join(output_dir, 'figure_network_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")


def draw_network(G, ax, title):
    """Draw network on axis with geographic layout."""
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(title, loc='left', fontweight='bold')
        return

    # Get positions
    pos = {c: COUNTRY_POS.get(c, (0, 0)) for c in G.nodes()}

    # Node colors by region
    node_colors = [REGION_COLORS.get(COUNTRY_REGIONS.get(c, 'Unknown'), 'gray')
                   for c in G.nodes()]

    # Edge weights for width
    edges = G.edges(data=True)
    edge_weights = [d['weight'] for _, _, d in edges]

    if edge_weights:
        edge_widths = [w * 2 for w in edge_weights]
    else:
        edge_widths = []

    # Draw
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                          node_size=200, alpha=0.9, edgecolors='white', linewidths=0.5)

    if edges:
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                              alpha=0.4, edge_color='gray')

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5, font_weight='bold')

    ax.set_title(title, loc='left', fontweight='bold')
    ax.axis('off')

    # Add legend
    legend_patches = [mpatches.Patch(color=c, label=r)
                      for r, c in list(REGION_COLORS.items())[:4]]
    ax.legend(handles=legend_patches, loc='lower left', fontsize=5, frameon=False)


def main():
    print("=" * 70)
    print("CORRELATION NETWORK EVOLUTION ANALYSIS")
    print("=" * 70)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    print("\nAnalyzing network evolution...")
    metrics_df = analyze_all_years(country_data, YEARS, EDGE_THRESHOLD)

    print("\n" + "-" * 50)
    print("NETWORK EVOLUTION SUMMARY")
    print("-" * 50)
    print(metrics_df[['year', 'n_nodes', 'n_edges', 'density', 'modularity', 'avg_clustering']].to_string(index=False))

    print("\nCreating figure...")
    create_network_figure(country_data, metrics_df, OUTPUT_DIR)

    print("\nSaving results...")
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'network_metrics_by_year.csv'), index=False)
    print(f"Saved: {RESULTS_DIR}/network_metrics_by_year.csv")

    print("\n" + "=" * 70)
    print("NETWORK ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
