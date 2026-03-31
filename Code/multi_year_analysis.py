# -*- coding: utf-8 -*-
"""
Multi-Year Wind Energy Correlation Analysis (2015-2024)

Performs year-by-year analysis of wind energy correlations and baseload
across European countries, tracking how patterns evolve over time.

Authors: Zajec & Kren, Jozef Stefan Institute
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os
import glob
from datetime import datetime

from correlation_utils import (
    pearson_with_ci,
    fdr_correction,
    compute_cv,
)

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "MultiYear")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Years to analyze
YEARS = list(range(2015, 2025))


# =============================================================================
# Data Loading
# =============================================================================

def load_all_country_data(data_dir):
    """
    Load all available country wind data files.

    Returns
    -------
    dict : country_code -> DataFrame with DatetimeIndex
    """
    country_data = {}

    files = glob.glob(os.path.join(data_dir, "*_wind_2015_2024.csv"))

    for filepath in files:
        filename = os.path.basename(filepath)
        country_code = filename.split('_')[0]

        try:
            df = pd.read_csv(filepath)

            # Parse datetime from first column (may have timezone info)
            date_col = df.columns[0]
            df['datetime'] = pd.to_datetime(df[date_col], utc=True)
            df = df.set_index('datetime')
            df = df.drop(columns=[date_col], errors='ignore')

            # Use Wind Total if available, otherwise sum columns
            if 'Wind Total' in df.columns:
                df['wind'] = df['Wind Total']
            else:
                # Sum numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df['wind'] = df[numeric_cols].fillna(0).sum(axis=1)

            country_data[country_code] = df[['wind']]
            print(f"  Loaded {country_code}: {len(df)} rows, {df.index.min().year}-{df.index.max().year}")

        except Exception as e:
            print(f"  Error loading {country_code}: {e}")

    return country_data


def get_year_data(country_data, year):
    """
    Extract data for a specific year from all countries.
    Resamples all data to hourly to ensure alignment.

    Returns
    -------
    dict : country_code -> numpy array of hourly wind values
    datetime_index : pandas DatetimeIndex for the year
    """
    year_data = {}
    resampled_dfs = {}

    for country, df in country_data.items():
        # Filter to year
        mask = df.index.year == year
        year_df = df[mask]

        if len(year_df) > 100:  # Minimum data threshold
            # Resample to hourly mean to align all countries
            hourly = year_df['wind'].resample('h').mean()
            hourly = hourly.dropna()
            if len(hourly) > 100:
                resampled_dfs[country] = hourly

    # Find common timestamps across all countries
    if resampled_dfs:
        common_index = None
        for country, series in resampled_dfs.items():
            if common_index is None:
                common_index = set(series.index)
            else:
                common_index = common_index.intersection(set(series.index))

        common_index = sorted(list(common_index))

        if len(common_index) > 100:
            datetime_index = pd.DatetimeIndex(common_index)
            for country, series in resampled_dfs.items():
                year_data[country] = series.loc[common_index].values

            return year_data, datetime_index

    return {}, None


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_year(country_data, year):
    """
    Perform complete analysis for a single year.

    Returns
    -------
    dict with keys: baseload, correlations, diversification
    """
    year_data, datetime_index = get_year_data(country_data, year)

    if len(year_data) < 3:
        return None

    countries = list(year_data.keys())
    n_countries = len(countries)

    # Calculate total European production
    n_samples = len(year_data[countries[0]])
    total_power = np.zeros(n_samples)
    for data in year_data.values():
        if len(data) == n_samples:
            total_power += np.nan_to_num(data)

    # Baseload analysis
    baseload = {
        'year': year,
        'n_countries': n_countries,
        'n_samples': n_samples,
        'min': np.min(total_power),
        'p1': np.percentile(total_power, 1),
        'p5': np.percentile(total_power, 5),
        'p10': np.percentile(total_power, 10),
        'mean': np.mean(total_power),
        'max': np.max(total_power),
        'std': np.std(total_power),
        'cv': compute_cv(total_power),
    }

    # Correlation analysis
    correlations = []
    for c1, c2 in combinations(countries, 2):
        data1 = year_data[c1]
        data2 = year_data[c2]

        if len(data1) != len(data2):
            continue

        result = pearson_with_ci(data1, data2)

        if result['n'] > 100:
            correlations.append({
                'year': year,
                'country1': c1,
                'country2': c2,
                'r': result['r'],
                'p': result['p'],
                'n': result['n'],
            })

    # Diversification analysis
    individual_cvs = []
    for country, data in year_data.items():
        cv = compute_cv(data)
        if not np.isnan(cv) and cv < 10:
            individual_cvs.append(cv)

    diversification = {
        'year': year,
        'n_countries': n_countries,
        'mean_individual_cv': np.mean(individual_cvs) if individual_cvs else np.nan,
        'aggregated_cv': baseload['cv'],
        'cv_reduction_pct': 0,
    }

    if individual_cvs and baseload['cv'] > 0:
        diversification['cv_reduction_pct'] = (
            (diversification['mean_individual_cv'] - baseload['cv']) /
            diversification['mean_individual_cv'] * 100
        )

    return {
        'baseload': baseload,
        'correlations': correlations,
        'diversification': diversification,
    }


def run_multi_year_analysis(country_data, years):
    """
    Run analysis for all specified years.
    """
    all_baseload = []
    all_correlations = []
    all_diversification = []

    for year in years:
        print(f"\nAnalyzing {year}...")
        result = analyze_year(country_data, year)

        if result:
            all_baseload.append(result['baseload'])
            all_correlations.extend(result['correlations'])
            all_diversification.append(result['diversification'])
            print(f"  Countries: {result['baseload']['n_countries']}, "
                  f"Min: {result['baseload']['min']:.0f} MW, "
                  f"CV: {result['baseload']['cv']:.3f}")
        else:
            print(f"  Insufficient data")

    return {
        'baseload': pd.DataFrame(all_baseload),
        'correlations': pd.DataFrame(all_correlations),
        'diversification': pd.DataFrame(all_diversification),
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_baseload_evolution(baseload_df, output_path):
    """Plot evolution of baseload statistics over years."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Min and percentiles
    ax1 = axes[0, 0]
    ax1.plot(baseload_df['year'], baseload_df['min']/1000, 'ro-', label='Minimum', linewidth=2, markersize=8)
    ax1.plot(baseload_df['year'], baseload_df['p5']/1000, 'b^-', label='5th percentile', linewidth=2)
    ax1.plot(baseload_df['year'], baseload_df['p10']/1000, 'gs-', label='10th percentile', linewidth=2)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Wind Production [GW]', fontsize=12)
    ax1.set_title('Baseload Evolution (Minimum and Low Percentiles)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(baseload_df['year'])

    # Mean and max
    ax2 = axes[0, 1]
    ax2.fill_between(baseload_df['year'], baseload_df['min']/1000, baseload_df['max']/1000,
                     alpha=0.3, label='Range')
    ax2.plot(baseload_df['year'], baseload_df['mean']/1000, 'g-', linewidth=3, label='Mean')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Wind Production [GW]', fontsize=12)
    ax2.set_title('Production Range Over Years', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(baseload_df['year'])

    # CV evolution
    ax3 = axes[1, 0]
    ax3.plot(baseload_df['year'], baseload_df['cv'], 'purple', linewidth=2, marker='o', markersize=8)
    ax3.set_xlabel('Year', fontsize=12)
    ax3.set_ylabel('Coefficient of Variation', fontsize=12)
    ax3.set_title('Variability (CV) Over Years', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(baseload_df['year'])

    # Number of countries
    ax4 = axes[1, 1]
    ax4.bar(baseload_df['year'], baseload_df['n_countries'], color='steelblue', alpha=0.7)
    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('Number of Countries', fontsize=12)
    ax4.set_title('Countries with Data', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks(baseload_df['year'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_correlation_evolution(corr_df, output_path):
    """Plot evolution of correlations over years."""
    if corr_df.empty:
        print("No correlation data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean correlation by year
    ax1 = axes[0]
    yearly_stats = corr_df.groupby('year')['r'].agg(['mean', 'std', 'median'])
    ax1.errorbar(yearly_stats.index, yearly_stats['mean'], yerr=yearly_stats['std'],
                 fmt='o-', capsize=5, linewidth=2, markersize=8, label='Mean ± Std')
    ax1.plot(yearly_stats.index, yearly_stats['median'], 's--', label='Median', alpha=0.7)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Mean Correlation Between Countries Over Time', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(yearly_stats.index)

    # Distribution of correlations by year (boxplot)
    ax2 = axes[1]
    years = sorted(corr_df['year'].unique())
    data_by_year = [corr_df[corr_df['year'] == y]['r'].values for y in years]
    bp = ax2.boxplot(data_by_year, labels=years, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Pearson Correlation', fontsize=12)
    ax2.set_title('Distribution of Pairwise Correlations by Year', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_diversification_evolution(div_df, output_path):
    """Plot diversification benefit over years."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(div_df['year'], div_df['mean_individual_cv'], 'ro-',
            label='Mean Individual Country CV', linewidth=2, markersize=8)
    ax.plot(div_df['year'], div_df['aggregated_cv'], 'bs-',
            label='Aggregated Portfolio CV', linewidth=2, markersize=8)

    # Shade the diversification benefit
    ax.fill_between(div_df['year'], div_df['aggregated_cv'], div_df['mean_individual_cv'],
                    alpha=0.3, color='green', label='Diversification Benefit')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('Diversification Benefit Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(div_df['year'])

    # Add percentage labels
    for _, row in div_df.iterrows():
        if not np.isnan(row['cv_reduction_pct']):
            ax.annotate(f"{row['cv_reduction_pct']:.0f}%",
                       xy=(row['year'], (row['mean_individual_cv'] + row['aggregated_cv'])/2),
                       ha='center', fontsize=9, color='green')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_key_pair_evolution(corr_df, output_path, pairs=None):
    """Plot correlation evolution for key country pairs."""
    if corr_df.empty:
        return

    # Default pairs to track
    if pairs is None:
        pairs = [
            ('BE', 'NL'), ('DE', 'NL'), ('DE', 'FR'), ('ES', 'PT'),
            ('AT', 'DE'), ('DE', 'PL'), ('FR', 'ES')
        ]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))

    for i, (c1, c2) in enumerate(pairs):
        # Find this pair in data (order might be reversed)
        mask = ((corr_df['country1'] == c1) & (corr_df['country2'] == c2)) | \
               ((corr_df['country1'] == c2) & (corr_df['country2'] == c1))
        pair_data = corr_df[mask].sort_values('year')

        if not pair_data.empty:
            ax.plot(pair_data['year'], pair_data['r'], 'o-',
                   color=colors[i], label=f'{c1}-{c2}', linewidth=2, markersize=6)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Correlation Evolution for Key Country Pairs', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary(results):
    """Print summary of multi-year analysis."""
    baseload_df = results['baseload']
    div_df = results['diversification']

    print("\n" + "=" * 70)
    print("MULTI-YEAR WIND ENERGY ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nBASELOAD EVOLUTION:")
    print("-" * 50)
    print(f"{'Year':<6} {'Countries':<10} {'Min [GW]':<12} {'Mean [GW]':<12} {'CV':<8}")
    print("-" * 50)
    for _, row in baseload_df.iterrows():
        print(f"{int(row['year']):<6} {int(row['n_countries']):<10} "
              f"{row['min']/1000:<12.2f} {row['mean']/1000:<12.2f} {row['cv']:<8.3f}")

    print("\nDIVERSIFICATION BENEFIT:")
    print("-" * 50)
    print(f"{'Year':<6} {'Indiv CV':<12} {'Aggreg CV':<12} {'Reduction':<10}")
    print("-" * 50)
    for _, row in div_df.iterrows():
        print(f"{int(row['year']):<6} {row['mean_individual_cv']:<12.3f} "
              f"{row['aggregated_cv']:<12.3f} {row['cv_reduction_pct']:<10.1f}%")

    # Trends
    if len(baseload_df) > 2:
        min_trend = np.polyfit(baseload_df['year'], baseload_df['min'], 1)[0]
        mean_trend = np.polyfit(baseload_df['year'], baseload_df['mean'], 1)[0]

        print(f"\nTRENDS:")
        print(f"  Minimum production trend: {min_trend/1000:+.2f} GW/year")
        print(f"  Mean production trend: {mean_trend/1000:+.2f} GW/year")

    print("\n" + "=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("MULTI-YEAR WIND ENERGY CORRELATION ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)

    if not country_data:
        print("ERROR: No data files found!")
        return

    print(f"\nLoaded {len(country_data)} countries")

    # Run analysis
    print("\n" + "-" * 40)
    print("Running year-by-year analysis...")
    results = run_multi_year_analysis(country_data, YEARS)

    # Save results
    print("\n" + "-" * 40)
    print("Saving results...")

    results['baseload'].to_csv(os.path.join(RESULTS_DIR, 'baseload_by_year.csv'), index=False)
    results['correlations'].to_csv(os.path.join(RESULTS_DIR, 'correlations_by_year.csv'), index=False)
    results['diversification'].to_csv(os.path.join(RESULTS_DIR, 'diversification_by_year.csv'), index=False)

    # Generate plots
    print("\n" + "-" * 40)
    print("Generating plots...")

    if not results['baseload'].empty:
        plot_baseload_evolution(results['baseload'],
                               os.path.join(FIGURE_DIR, 'baseload_evolution.png'))

    if not results['correlations'].empty:
        plot_correlation_evolution(results['correlations'],
                                  os.path.join(FIGURE_DIR, 'correlation_evolution.png'))
        plot_key_pair_evolution(results['correlations'],
                               os.path.join(FIGURE_DIR, 'key_pairs_evolution.png'))

    if not results['diversification'].empty:
        plot_diversification_evolution(results['diversification'],
                                       os.path.join(FIGURE_DIR, 'diversification_evolution.png'))

    # Print summary
    print_summary(results)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
