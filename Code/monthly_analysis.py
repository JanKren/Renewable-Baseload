# -*- coding: utf-8 -*-
"""
Monthly Wind Energy Analysis (2015-2024)

Analyzes wind energy patterns month-by-month within each year.

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime

from correlation_utils import compute_cv

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Monthly")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

YEARS = list(range(2015, 2025))
MONTHS = list(range(1, 13))
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def load_all_country_data(data_dir):
    """Load all country wind data files."""
    country_data = {}

    files = glob.glob(os.path.join(data_dir, "*_wind_2015_2024.csv"))

    for filepath in files:
        filename = os.path.basename(filepath)
        country_code = filename.split('_')[0]

        try:
            df = pd.read_csv(filepath)

            # Parse datetime
            date_col = df.columns[0]
            df['datetime'] = pd.to_datetime(df[date_col], utc=True)
            df = df.set_index('datetime')
            df = df.drop(columns=[date_col], errors='ignore')

            # Get wind total
            if 'Wind Total' in df.columns:
                df['wind'] = df['Wind Total']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df['wind'] = df[numeric_cols].fillna(0).sum(axis=1)

            country_data[country_code] = df[['wind']]

        except Exception as e:
            print(f"  Error loading {country_code}: {e}")

    return country_data


def analyze_month(country_data, year, month):
    """Analyze a specific month."""
    month_data = {}

    for country, df in country_data.items():
        mask = (df.index.year == year) & (df.index.month == month)
        month_df = df[mask]

        if len(month_df) > 24:  # At least 1 day of hourly data
            # Resample to hourly
            hourly = month_df['wind'].resample('h').mean().dropna()
            if len(hourly) > 24:
                month_data[country] = hourly

    if len(month_data) < 3:
        return None

    # Find common timestamps
    common_index = None
    for country, series in month_data.items():
        if common_index is None:
            common_index = set(series.index)
        else:
            common_index = common_index.intersection(set(series.index))

    common_index = sorted(list(common_index))

    if len(common_index) < 24:
        return None

    # Align data
    aligned_data = {}
    for country, series in month_data.items():
        aligned_data[country] = series.loc[common_index].values

    # Calculate total
    n_samples = len(common_index)
    total_power = np.zeros(n_samples)
    for data in aligned_data.values():
        total_power += np.nan_to_num(data)

    # Calculate metrics
    individual_cvs = [compute_cv(data) for data in aligned_data.values()
                      if not np.isnan(compute_cv(data)) and compute_cv(data) < 10]

    result = {
        'year': year,
        'month': month,
        'month_name': MONTH_NAMES[month - 1],
        'n_countries': len(aligned_data),
        'n_samples': n_samples,
        'min': np.min(total_power),
        'max': np.max(total_power),
        'mean': np.mean(total_power),
        'std': np.std(total_power),
        'cv_aggregate': compute_cv(total_power),
        'cv_individual_mean': np.mean(individual_cvs) if individual_cvs else np.nan,
        'cv_reduction': (1 - compute_cv(total_power) / np.mean(individual_cvs)) * 100 if individual_cvs else np.nan
    }

    return result


def run_monthly_analysis(country_data):
    """Run analysis for all months in all years."""
    results = []

    for year in YEARS:
        print(f"\nAnalyzing {year}...")
        for month in MONTHS:
            result = analyze_month(country_data, year, month)
            if result:
                results.append(result)
                print(f"  {MONTH_NAMES[month-1]}: Min={result['min']/1000:.1f} GW, CV={result['cv_aggregate']:.3f}")

    return pd.DataFrame(results)


def plot_monthly_heatmap(df, output_dir):
    """Create heatmap of monthly metrics."""

    # Pivot for heatmap
    metrics = ['min', 'mean', 'cv_aggregate', 'cv_reduction']
    titles = ['Minimum Production [GW]', 'Mean Production [GW]',
              'Coefficient of Variation', 'CV Reduction [%]']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = df.pivot(index='month', columns='year', values=metric)

        # Convert to GW for power metrics
        if metric in ['min', 'mean']:
            pivot = pivot / 1000

        sns.heatmap(pivot, ax=ax, cmap='viridis', annot=True, fmt='.1f',
                    yticklabels=MONTH_NAMES)
        ax.set_title(title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Month')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monthly_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/monthly_heatmaps.png")


def plot_seasonal_patterns(df, output_dir):
    """Plot seasonal patterns aggregated across years."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average by month
    monthly_avg = df.groupby('month').agg({
        'min': 'mean',
        'mean': 'mean',
        'cv_aggregate': 'mean',
        'cv_reduction': 'mean'
    })

    # Plot 1: Min and Mean production
    ax1 = axes[0, 0]
    ax1.bar(np.arange(12) - 0.2, monthly_avg['min'] / 1000, 0.4, label='Minimum', color='steelblue')
    ax1.bar(np.arange(12) + 0.2, monthly_avg['mean'] / 1000, 0.4, label='Mean', color='coral')
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(MONTH_NAMES)
    ax1.set_ylabel('Production [GW]')
    ax1.set_title('Seasonal Wind Production Pattern (2015-2024 Average)')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: CV by month
    ax2 = axes[0, 1]
    ax2.plot(range(12), monthly_avg['cv_aggregate'], 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xticks(range(12))
    ax2.set_xticklabels(MONTH_NAMES)
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('Variability by Month')
    ax2.grid(alpha=0.3)

    # Plot 3: CV reduction by month
    ax3 = axes[1, 0]
    ax3.bar(range(12), monthly_avg['cv_reduction'], color='purple', alpha=0.7)
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(MONTH_NAMES)
    ax3.set_ylabel('CV Reduction [%]')
    ax3.set_title('Diversification Benefit by Month')
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Year-over-year by month (min production)
    ax4 = axes[1, 1]
    for year in YEARS:
        year_data = df[df['year'] == year]
        ax4.plot(year_data['month'] - 1, year_data['min'] / 1000, 'o-',
                 label=str(year), alpha=0.7, linewidth=1.5)
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(MONTH_NAMES)
    ax4.set_ylabel('Minimum Production [GW]')
    ax4.set_title('Monthly Minimum by Year')
    ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonal_patterns.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/seasonal_patterns.png")


def print_summary(df):
    """Print summary statistics."""

    print("\n" + "=" * 70)
    print("MONTHLY ANALYSIS SUMMARY")
    print("=" * 70)

    # Best and worst months
    print("\nBEST MONTHS (Highest Minimum Production):")
    best = df.nlargest(5, 'min')[['year', 'month_name', 'min', 'mean', 'cv_aggregate']]
    best['min'] = best['min'] / 1000
    best['mean'] = best['mean'] / 1000
    print(best.to_string(index=False))

    print("\nWORST MONTHS (Lowest Minimum Production):")
    worst = df.nsmallest(5, 'min')[['year', 'month_name', 'min', 'mean', 'cv_aggregate']]
    worst['min'] = worst['min'] / 1000
    worst['mean'] = worst['mean'] / 1000
    print(worst.to_string(index=False))

    # Seasonal averages
    print("\nSEASONAL AVERAGES:")
    seasons = {
        'Winter (DJF)': [12, 1, 2],
        'Spring (MAM)': [3, 4, 5],
        'Summer (JJA)': [6, 7, 8],
        'Autumn (SON)': [9, 10, 11]
    }

    for season, months in seasons.items():
        season_data = df[df['month'].isin(months)]
        print(f"  {season}:")
        print(f"    Mean production: {season_data['mean'].mean()/1000:.1f} GW")
        print(f"    Min production:  {season_data['min'].mean()/1000:.1f} GW")
        print(f"    CV aggregate:    {season_data['cv_aggregate'].mean():.3f}")
        print(f"    CV reduction:    {season_data['cv_reduction'].mean():.1f}%")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("MONTHLY WIND ENERGY ANALYSIS (2015-2024)")
    print("=" * 70)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    print("\nRunning monthly analysis...")
    results_df = run_monthly_analysis(country_data)

    # Save results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'monthly_statistics.csv'), index=False)
    print(f"\nSaved: {RESULTS_DIR}/monthly_statistics.csv")

    # Generate plots
    print("\nGenerating plots...")
    plot_monthly_heatmap(results_df, FIGURE_DIR)
    plot_seasonal_patterns(results_df, FIGURE_DIR)

    # Print summary
    print_summary(results_df)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
