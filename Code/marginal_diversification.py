# -*- coding: utf-8 -*-
"""
Marginal Diversification Value Analysis

Ranks countries by their marginal contribution to portfolio CV reduction.
Provides actionable guidance for interconnection investment prioritization.

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Diversification")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Average backup capacity cost (EUR/kW)
BACKUP_COST_EUR_PER_KW = 900  # Gas peaker plant

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
    'gradient': plt.cm.RdYlGn_r,
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


def compute_aligned_data(country_data, year=2024):
    """Compute aligned hourly data for a single year."""

    year_data = {}

    for country, df in country_data.items():
        year_df = df[df.index.year == year]
        if len(year_df) > 1000:
            hourly = year_df['wind'].resample('h').mean().dropna()
            if len(hourly) > 1000:
                year_data[country] = hourly

    # Find common index
    common_idx = None
    for country, series in year_data.items():
        if common_idx is None:
            common_idx = set(series.index)
        else:
            common_idx = common_idx.intersection(set(series.index))

    common_idx = sorted(list(common_idx))

    # Create aligned DataFrame
    aligned_df = pd.DataFrame({c: year_data[c].loc[common_idx].values
                               for c in year_data.keys()})

    return aligned_df


def compute_aligned_data_multiyear(country_data, years=range(2015, 2025)):
    """Compute aligned hourly data across multiple years.

    For each year, aligns countries that have data, then concatenates
    all years. Countries that don't appear in all years are still included
    for the years they have data — the greedy algorithm uses the full
    concatenated timeseries so countries with more data naturally
    contribute more to the CV calculation.

    To keep it fair, we only include countries present in at least 5 years.
    """

    all_frames = []
    country_year_count = {}

    for year in years:
        year_data = {}
        for country, df in country_data.items():
            year_df = df[df.index.year == year]
            if len(year_df) > 1000:
                hourly = year_df['wind'].resample('h').mean().dropna()
                if len(hourly) > 1000:
                    year_data[country] = hourly
                    country_year_count[country] = country_year_count.get(country, 0) + 1

        if not year_data:
            continue

        # Find common index for this year
        common_idx = None
        for country, series in year_data.items():
            if common_idx is None:
                common_idx = set(series.index)
            else:
                common_idx = common_idx.intersection(set(series.index))

        common_idx = sorted(list(common_idx))
        if len(common_idx) < 1000:
            continue

        year_df = pd.DataFrame({c: year_data[c].loc[common_idx].values
                                for c in year_data.keys()},
                               index=common_idx)
        all_frames.append(year_df)

    # Concatenate all years
    combined = pd.concat(all_frames, axis=0)

    # Keep only countries present in >= 5 years
    keep = [c for c, n in country_year_count.items() if n >= 5]
    combined = combined[keep].dropna()

    return combined


def compute_cv(series):
    """Compute coefficient of variation."""
    if series.std() == 0 or series.mean() == 0:
        return np.nan
    return series.std() / series.mean()


def greedy_portfolio_construction(aligned_df, start_country=None):
    """
    Build portfolio greedily by always adding the country
    that provides maximum marginal CV reduction.

    Parameters:
        aligned_df: DataFrame with countries as columns, hourly data as rows
        start_country: country code to start from (default: largest producer)
    """

    countries = list(aligned_df.columns)
    n_countries = len(countries)

    # Start with specified country or largest producer
    if start_country is None:
        country_means = aligned_df.mean()
        start_country = country_means.idxmax()
    elif start_country not in countries:
        raise ValueError(f"start_country '{start_country}' not in data columns")

    portfolio = [start_country]
    remaining = [c for c in countries if c != start_country]

    # Track portfolio evolution
    evolution = [{
        'step': 1,
        'country_added': start_country,
        'portfolio_size': 1,
        'portfolio_cv': compute_cv(aligned_df[start_country]),
        'marginal_cv_reduction': 0,
        'portfolio_mean_gw': aligned_df[start_country].mean() / 1000,
        'portfolio_min_gw': aligned_df[start_country].min() / 1000
    }]

    step = 1

    while remaining:
        step += 1
        best_country = None
        best_cv = float('inf')

        current_portfolio_sum = aligned_df[portfolio].sum(axis=1)
        current_cv = compute_cv(current_portfolio_sum)

        for candidate in remaining:
            new_sum = current_portfolio_sum + aligned_df[candidate]
            new_cv = compute_cv(new_sum)

            if new_cv < best_cv:
                best_cv = new_cv
                best_country = candidate

        portfolio.append(best_country)
        remaining.remove(best_country)

        new_portfolio_sum = aligned_df[portfolio].sum(axis=1)

        evolution.append({
            'step': step,
            'country_added': best_country,
            'portfolio_size': len(portfolio),
            'portfolio_cv': best_cv,
            'marginal_cv_reduction': (current_cv - best_cv) / current_cv * 100 if current_cv > 0 else 0,
            'portfolio_mean_gw': new_portfolio_sum.mean() / 1000,
            'portfolio_min_gw': new_portfolio_sum.min() / 1000
        })

    return pd.DataFrame(evolution), portfolio


def compute_economic_value(evolution_df):
    """Compute economic value of diversification."""

    # Total CV reduction
    initial_cv = evolution_df.iloc[0]['portfolio_cv']
    final_cv = evolution_df.iloc[-1]['portfolio_cv']
    total_cv_reduction = (initial_cv - final_cv) / initial_cv * 100

    # Avoided backup capacity (rough estimate)
    # CV reduction means less variability, hence less backup needed
    # Use final portfolio mean as base
    final_mean_gw = evolution_df.iloc[-1]['portfolio_mean_gw']

    # Assume backup needed = CV * mean * some factor
    # This is a simplified estimate
    backup_reduction_gw = (initial_cv - final_cv) * final_mean_gw * 0.5

    backup_value_billion_eur = backup_reduction_gw * 1e6 * BACKUP_COST_EUR_PER_KW / 1e9

    return {
        'total_cv_reduction_pct': total_cv_reduction,
        'backup_reduction_gw': backup_reduction_gw,
        'backup_value_billion_eur': backup_value_billion_eur
    }


def create_marginal_value_figure(evolution_df, economic_value, output_dir):
    """Create figure showing marginal diversification value."""

    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.8))

    # Panel A: CV vs Portfolio Size
    ax1 = axes[0]
    # Color points by whether they reduce or increase CV
    marginal = evolution_df['marginal_cv_reduction'].values
    sizes = evolution_df['portfolio_size'].values
    cvs = evolution_df['portfolio_cv'].values

    # Plot line
    ax1.plot(sizes, cvs, '-', color='gray', linewidth=1, zorder=1)
    # Beneficial additions (green) vs harmful (red)
    beneficial = marginal >= 0
    ax1.scatter(sizes[beneficial], cvs[beneficial], c=COLORS['primary'],
                s=20, zorder=2, label='CV decreases')
    ax1.scatter(sizes[~beneficial], cvs[~beneficial], c=COLORS['secondary'],
                s=20, zorder=2, label='CV increases')

    # Annotate key countries
    for _, row in evolution_df.iterrows():
        if row['country_added'] in ['DE', 'ES', 'SE', 'IT', 'GB', 'NL', 'FR']:
            offset = (5, 5) if row['marginal_cv_reduction'] >= 0 else (5, -10)
            ax1.annotate(row['country_added'],
                         xy=(row['portfolio_size'], row['portfolio_cv']),
                         xytext=offset, textcoords='offset points',
                         fontsize=6, color='gray')

    ax1.set_xlabel('Number of Countries in Portfolio')
    ax1.set_ylabel('Portfolio CV')
    ax1.set_title('a) Portfolio CV vs size', loc='left', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=6, frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Marginal CV Reduction by Country
    ax2 = axes[1]
    colors_bar = [COLORS['primary'] if m >= 0 else COLORS['secondary']
                  for m in evolution_df['marginal_cv_reduction']]
    ax2.bar(range(len(evolution_df)), evolution_df['marginal_cv_reduction'],
            color=colors_bar, edgecolor='white', linewidth=0.5)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax2.set_ylabel('Marginal CV Reduction [%]')
    ax2.set_title('b) Marginal diversification value', loc='left', fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Use country codes as x-axis tick labels
    ax2.set_xticks(range(len(evolution_df)))
    ax2.set_xticklabels(evolution_df['country_added'], rotation=90, fontsize=5, ha='center')

    # Panel C: Portfolio Mean and Min
    ax3 = axes[2]
    ax3.plot(evolution_df['portfolio_size'], evolution_df['portfolio_mean_gw'],
             'o-', color=COLORS['primary'], label='Mean', linewidth=2, markersize=3)
    ax3.plot(evolution_df['portfolio_size'], evolution_df['portfolio_min_gw'],
             's--', color=COLORS['secondary'], label='Minimum', linewidth=2, markersize=3)
    ax3.set_xlabel('Number of Countries')
    ax3.set_ylabel('Production [GW]')
    ax3.set_title('c) Portfolio production', loc='left', fontweight='bold')
    ax3.legend(loc='center right', frameon=False, fontsize=6)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'figure_marginal_diversification.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")


def sensitivity_analysis(aligned_df, start_countries=None):
    """
    Run greedy portfolio construction from multiple starting countries
    to test ranking robustness.

    Parameters:
        aligned_df: DataFrame with countries as columns
        start_countries: list of country codes to start from
                        (default: DE, FR, ES, GB)

    Returns:
        dict of {start_country: (evolution_df, portfolio_order)}
        summary DataFrame comparing rankings
    """
    if start_countries is None:
        start_countries = ['DE', 'FR', 'ES', 'GB']

    # Filter to countries present in data
    start_countries = [c for c in start_countries if c in aligned_df.columns]

    results = {}
    for sc in start_countries:
        print(f"\n  Starting from {sc}...")
        evo_df, order = greedy_portfolio_construction(aligned_df, start_country=sc)
        results[sc] = (evo_df, order)

        # Print top-5
        top5 = evo_df.head(6)  # includes start country
        countries_str = ' → '.join(top5['country_added'].values)
        print(f"    Top-6: {countries_str}")
        print(f"    Final CV: {evo_df.iloc[-1]['portfolio_cv']:.3f}")

    # Build comparison table: for each starting country, show rank of key countries
    comparison_rows = []
    key_countries = ['DE', 'ES', 'SE', 'IT', 'GR', 'PT', 'FR', 'GB', 'IE', 'NO']

    for sc, (evo_df, order) in results.items():
        row = {'start_country': sc}
        for kc in key_countries:
            if kc in order:
                rank = order.index(kc) + 1
                row[f'rank_{kc}'] = rank
            else:
                row[f'rank_{kc}'] = np.nan
        # Also record final portfolio CV (should be same for all)
        row['final_cv'] = evo_df.iloc[-1]['portfolio_cv']
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    return results, comparison_df


def create_sensitivity_figure(sensitivity_results, output_dir):
    """Create SI figure showing CV curves from different starting countries."""

    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.0),
                              gridspec_kw={'wspace': 0.5, 'width_ratios': [1.2, 1]})

    # Panel A: CV curves from different starting countries
    ax1 = axes[0]
    colors_start = {'DE': '#2166ac', 'FR': '#b2182b', 'ES': '#d6604d', 'GB': '#4393c3'}

    for sc, (evo_df, order) in sensitivity_results.items():
        color = colors_start.get(sc, 'gray')
        ax1.plot(evo_df['portfolio_size'], evo_df['portfolio_cv'],
                 'o-', color=color, linewidth=1.5, markersize=2.5,
                 label=f'Start: {sc}', alpha=0.8)

    ax1.set_xlabel('Number of Countries in Portfolio')
    ax1.set_ylabel('Portfolio CV')
    ax1.set_title('a) CV vs. starting country',
                   loc='left', fontweight='bold')
    ax1.legend(loc='upper right', frameon=False, fontsize=6)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Rank comparison heatmap
    ax2 = axes[1]

    start_countries = list(sensitivity_results.keys())
    # Get all countries from first result
    first_evo = list(sensitivity_results.values())[0][0]
    all_countries = first_evo['country_added'].values

    # Build rank matrix (top 10 only for readability)
    show_countries = []
    for sc, (evo_df, order) in sensitivity_results.items():
        show_countries.extend(order[:8])
    show_countries = list(dict.fromkeys(show_countries))[:12]  # unique, up to 12

    rank_matrix = np.full((len(show_countries), len(start_countries)), np.nan)
    for j, sc in enumerate(start_countries):
        evo_df, order = sensitivity_results[sc]
        for i, kc in enumerate(show_countries):
            if kc in order:
                rank_matrix[i, j] = order.index(kc) + 1

    im = ax2.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto',
                     vmin=1, vmax=len(all_countries))

    ax2.set_xticks(range(len(start_countries)))
    ax2.set_xticklabels([f'Start: {sc}' for sc in start_countries], fontsize=6)
    ax2.set_yticks(range(len(show_countries)))
    ax2.set_yticklabels(show_countries, fontsize=7)
    ax2.set_title('b) Rank comparison', loc='left', fontweight='bold')

    # Add rank text
    for i in range(len(show_countries)):
        for j in range(len(start_countries)):
            if not np.isnan(rank_matrix[i, j]):
                text_color = 'white' if rank_matrix[i, j] > 15 else 'black'
                ax2.text(j, i, f'{int(rank_matrix[i, j])}',
                         ha='center', va='center', fontsize=6, color=text_color)

    plt.colorbar(im, ax=ax2, shrink=0.7, label='Rank')

    output_path = os.path.join(output_dir, 'figure_sensitivity_start_country.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")


def main():
    print("=" * 70)
    print("MARGINAL DIVERSIFICATION VALUE ANALYSIS")
    print("=" * 70)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    print("\nComputing aligned data (full decade 2015-2024)...")
    aligned_df = compute_aligned_data_multiyear(country_data, years=range(2015, 2025))
    print(f"Aligned {len(aligned_df.columns)} countries, {len(aligned_df)} hours")

    print("\nBuilding optimal portfolio greedily...")
    evolution_df, portfolio_order = greedy_portfolio_construction(aligned_df)

    print("\n" + "-" * 50)
    print("OPTIMAL PORTFOLIO CONSTRUCTION ORDER")
    print("-" * 50)
    for _, row in evolution_df.iterrows():
        print(f"  {row['step']:2d}. {row['country_added']:3s}: "
              f"CV={row['portfolio_cv']:.3f}, "
              f"Marginal reduction={row['marginal_cv_reduction']:.1f}%")

    print("\nComputing economic value...")
    economic_value = compute_economic_value(evolution_df)
    print(f"Total CV reduction: {economic_value['total_cv_reduction_pct']:.1f}%")
    print(f"Equivalent avoided backup capacity: ~{economic_value['backup_reduction_gw']:.1f} GW")
    print(f"Estimated economic value: ~€{economic_value['backup_value_billion_eur']:.0f}B")

    print("\nCreating figure...")
    create_marginal_value_figure(evolution_df, economic_value, OUTPUT_DIR)

    print("\nSaving results...")
    evolution_df.to_csv(os.path.join(RESULTS_DIR, 'marginal_diversification.csv'), index=False)
    print(f"Saved: {RESULTS_DIR}/marginal_diversification.csv")

    # --- Sensitivity analysis (A1) ---
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: STARTING COUNTRY")
    print("=" * 70)

    sensitivity_results, comparison_df = sensitivity_analysis(
        aligned_df, start_countries=['DE', 'FR', 'ES', 'GB'])

    print("\n" + "-" * 50)
    print("RANK COMPARISON TABLE")
    print("-" * 50)
    print(comparison_df.to_string(index=False))

    # Save sensitivity results
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'sensitivity_start_country.csv'), index=False)
    print(f"\nSaved: {RESULTS_DIR}/sensitivity_start_country.csv")

    # Save individual evolution CSVs
    for sc, (evo_df, order) in sensitivity_results.items():
        evo_df.to_csv(os.path.join(RESULTS_DIR, f'marginal_diversification_start_{sc}.csv'),
                       index=False)

    # Create sensitivity figure
    print("\nCreating sensitivity figure...")
    create_sensitivity_figure(sensitivity_results, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("MARGINAL DIVERSIFICATION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
