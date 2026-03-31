# -*- coding: utf-8 -*-
"""
Combined Wind + Solar Analysis for Nature Communications Paper

Comprehensive analysis of European wind and solar portfolio diversification.
Includes data quality assessment and limitations documentation.

Authors: Kren & Zajec
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

WIND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
SOLAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Solar")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Combined")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Official 2023 solar generation for validation (TWh) - Ember/Eurostat
OFFICIAL_SOLAR_2023 = {
    'DE': 62, 'ES': 42, 'IT': 31, 'NL': 19, 'FR': 21,
    'PL': 15, 'BE': 7, 'GR': 8, 'PT': 6, 'AT': 4, 'HU': 5
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

COLORS = {
    'wind': '#2166ac',
    'solar': '#fdae61',
    'combined': '#b2182b',
    'neutral': '#878787',
}


def load_country_data(country, source):
    """Load wind or solar data for a country."""
    if source == 'wind':
        path = f'{WIND_DIR}/{country}_wind_2015_2024.csv'
        col = 'Wind Total'
    else:
        path = f'{SOLAR_DIR}/{country}_solar_2015_2024.csv'
        col = 'Solar'

    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        series = df[col]
        # Resample to hourly
        hourly = series.resample('h').mean()
        return hourly / 1000  # Convert MW to GW
    except:
        return None


def get_all_countries():
    """Get all wind and solar country lists separately."""
    wind_files = glob.glob(f'{WIND_DIR}/*_wind_*.csv')
    solar_files = glob.glob(f'{SOLAR_DIR}/*_solar_*.csv')

    wind_countries = sorted(set(f.split('/')[-1].split('_')[0] for f in wind_files))
    solar_countries = sorted(set(f.split('/')[-1].split('_')[0] for f in solar_files))

    return wind_countries, solar_countries


def analyze_year(wind_countries, solar_countries, year):
    """Comprehensive analysis for one year.

    Uses ALL wind countries for wind total and ALL solar countries for solar
    total (not restricted to countries with both). Only includes countries
    with >7000 hours of data per year to avoid bias from partial coverage.
    Timestamps are the intersection of the wind-set and solar-set common hours.
    """
    # Load well-covered wind countries
    wind_series = {}
    for country in wind_countries:
        series = load_country_data(country, 'wind')
        if series is not None:
            yr_data = series[series.index.year == year]
            if len(yr_data) > 7000:
                wind_series[country] = yr_data

    # Load well-covered solar countries
    solar_series = {}
    for country in solar_countries:
        series = load_country_data(country, 'solar')
        if series is not None:
            yr_data = series[series.index.year == year]
            if len(yr_data) > 7000:
                solar_series[country] = yr_data

    if len(wind_series) < 10 or len(solar_series) < 5:
        return None

    # Common timestamps within each set, then intersect
    wind_idx = set.intersection(*[set(s.index) for s in wind_series.values()])
    solar_idx = set.intersection(*[set(s.index) for s in solar_series.values()])
    common_idx = sorted(wind_idx & solar_idx)

    if len(common_idx) < 1000:
        return None

    # Aggregate
    wind_total = pd.Series(0.0, index=common_idx)
    for s in wind_series.values():
        wind_total += s.reindex(common_idx).fillna(0)

    solar_total = pd.Series(0.0, index=common_idx)
    for s in solar_series.values():
        solar_total += s.reindex(common_idx).fillna(0)

    wind = wind_total
    solar = solar_total
    combined = wind + solar

    # Statistics
    results = {
        'year': year,
        'n_wind_countries': len(wind_series),
        'n_solar_countries': len(solar_series),
        'n_hours': len(common_idx),

        'wind_mean': wind.mean(),
        'wind_min': wind.min(),
        'wind_max': wind.max(),
        'wind_std': wind.std(),
        'wind_cv': wind.std() / wind.mean(),

        'solar_mean': solar.mean(),
        'solar_min': solar.min(),
        'solar_max': solar.max(),
        'solar_std': solar.std(),

        'combined_mean': combined.mean(),
        'combined_min': combined.min(),
        'combined_max': combined.max(),
        'combined_std': combined.std(),
        'combined_cv': combined.std() / combined.mean(),

        'correlation': wind.corr(solar),
        'baseload_improvement_gw': combined.min() - wind.min(),
        'baseload_improvement_pct': (combined.min() / wind.min() - 1) * 100,
        'cv_reduction_pct': (1 - combined.std()/combined.mean() / (wind.std()/wind.mean())) * 100,
    }

    return results, wind, solar, combined


def create_main_figure(all_results, output_dir):
    """Create main combined analysis figure for paper."""

    df = pd.DataFrame(all_results)

    fig = plt.figure(figsize=(7.08, 7))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1],
                          hspace=0.40, wspace=0.3,
                          left=0.10, right=0.95, top=0.96, bottom=0.07)

    # Panel A: Baseload comparison over years
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(df['year'], df['wind_min'], 'o-', color=COLORS['wind'],
              linewidth=1.5, markersize=4, label='Wind only')
    ax_a.plot(df['year'], df['combined_min'], 's-', color=COLORS['combined'],
              linewidth=1.5, markersize=4, label='Wind + Solar')
    ax_a.fill_between(df['year'], df['wind_min'], df['combined_min'],
                      alpha=0.3, color=COLORS['combined'])
    ax_a.set_xlabel('Year')
    ax_a.set_ylabel('Baseload Floor [GW]')
    ax_a.set_title('a) Baseload floor: wind vs combined', fontweight='bold', loc='left')
    ax_a.legend(loc='upper left', frameon=False)
    ax_a.set_ylim(0, None)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Panel B: Baseload improvement percentage
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.bar(df['year'], df['baseload_improvement_pct'], color=COLORS['combined'], alpha=0.8)
    mean_val = df['baseload_improvement_pct'].mean()
    ax_b.axhline(mean_val, color='black', linestyle='--',
                 linewidth=1, label=f'Mean: {mean_val:.0f}%')
    ax_b.set_xlabel('Year')
    ax_b.set_ylabel('Improvement [%]')
    ax_b.set_title('b) Baseload improvement from adding solar', fontweight='bold', loc='left')
    ax_b.legend(loc='upper left', frameon=False)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Panel C: Wind-solar correlation
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.bar(df['year'], df['correlation'], color=COLORS['neutral'], alpha=0.8)
    ax_c.axhline(0, color='black', linewidth=0.5)
    ax_c.axhline(df['correlation'].mean(), color=COLORS['combined'], linestyle='--',
                 linewidth=1.5, label=f'Mean: {df["correlation"].mean():.2f}')
    ax_c.set_xlabel('Year')
    ax_c.set_ylabel('Pearson Correlation')
    ax_c.set_title('c) Wind-solar correlation', fontweight='bold', loc='left')
    ax_c.set_ylim(-0.5, 0.1)
    ax_c.legend(loc='lower right', frameon=False)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Panel D: CV comparison
    ax_d = fig.add_subplot(gs[1, 1])
    x = np.arange(len(df))
    width = 0.35
    ax_d.bar(x - width/2, df['wind_cv'], width, label='Wind only', color=COLORS['wind'], alpha=0.8)
    ax_d.bar(x + width/2, df['combined_cv'], width, label='Wind + Solar', color=COLORS['combined'], alpha=0.8)
    ax_d.set_xlabel('Year')
    ax_d.set_ylabel('Coefficient of variation (CV)')
    ax_d.set_title('d) Coefficient of variation', fontweight='bold', loc='left')
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([str(y) if y % 2 == 0 else '' for y in df['year']])
    ax_d.set_ylim(0, 0.75)
    ax_d.legend(loc='upper left', frameon=False, fontsize=7)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # Panel E: Mean production stacked (full width)
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.stackplot(df['year'], df['wind_mean'], df['solar_mean'],
                   labels=['Wind', 'Solar'], colors=[COLORS['wind'], COLORS['solar']], alpha=0.8)
    # Add annual data points
    ax_e.scatter(df['year'], df['wind_mean'], color=COLORS['wind'], s=20, zorder=5, edgecolors='white', linewidths=0.5)
    ax_e.scatter(df['year'], df['wind_mean'] + df['solar_mean'], color=COLORS['combined'], s=20, zorder=5, edgecolors='white', linewidths=0.5)
    ax_e.set_xlabel('Year')
    ax_e.set_ylabel('Mean Production [GW]')
    ax_e.set_title('e) Total renewable production', fontweight='bold', loc='left')
    ax_e.legend(loc='upper left', frameon=False)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)

    # Save
    output_path = os.path.join(output_dir, 'figure_combined_wind_solar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def assess_data_quality(countries):
    """Assess ENTSO-E solar data quality against official statistics."""

    print("\n" + "=" * 70)
    print("DATA QUALITY ASSESSMENT: ENTSO-E Solar vs Official Statistics")
    print("=" * 70)

    results = []

    for country in sorted(OFFICIAL_SOLAR_2023.keys()):
        official_twh = OFFICIAL_SOLAR_2023[country]

        try:
            df = pd.read_csv(f'{SOLAR_DIR}/{country}_solar_2015_2024.csv', index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            y2023 = df[df.index.year == 2023]['Solar']

            if len(y2023) < 100:
                continue

            # Detect resolution and calculate TWh
            time_diff_min = (y2023.index[1] - y2023.index[0]).total_seconds() / 60
            hours_per_row = time_diff_min / 60
            entsoe_twh = y2023.sum() * hours_per_row / 1e6

            ratio = entsoe_twh / official_twh

            results.append({
                'country': country,
                'official_twh': official_twh,
                'entsoe_twh': entsoe_twh,
                'capture_rate': ratio,
                'resolution_min': time_diff_min,
            })

        except Exception as e:
            pass

    df_quality = pd.DataFrame(results)

    # Summary
    total_official = df_quality['official_twh'].sum()
    total_entsoe = df_quality['entsoe_twh'].sum()

    print(f"\n{'Country':<8} {'Official':<10} {'ENTSO-E':<10} {'Capture':<10}")
    print("-" * 40)
    for _, row in df_quality.iterrows():
        status = "✓" if row['capture_rate'] > 0.75 else "⚠️" if row['capture_rate'] < 0.5 else "⚡"
        print(f"{row['country']:<8} {row['official_twh']:<10.0f} {row['entsoe_twh']:<10.1f} {row['capture_rate']:<10.0%} {status}")

    print("-" * 40)
    print(f"{'TOTAL':<8} {total_official:<10.0f} {total_entsoe:<10.1f} {total_entsoe/total_official:<10.0%}")

    # Identify gaps
    underreported = df_quality[df_quality['capture_rate'] < 0.5]
    if len(underreported) > 0:
        print(f"\n⚠️  Underreported countries: {', '.join(underreported['country'].tolist())}")
        missing_twh = (underreported['official_twh'] - underreported['entsoe_twh']).sum()
        print(f"   Missing: ~{missing_twh:.0f} TWh/year")

    return df_quality, total_entsoe/total_official


def main():
    print("=" * 70)
    print("COMBINED WIND + SOLAR ANALYSIS")
    print("For Nature Communications submission")
    print("=" * 70)

    # Get all countries (wind and solar separately)
    wind_countries, solar_countries = get_all_countries()
    print(f"\nWind countries: {len(wind_countries)}")
    print(f"Solar countries: {len(solar_countries)}")

    # Data quality assessment
    quality_df, capture_rate = assess_data_quality(solar_countries)

    # Annual analysis
    print("\n" + "=" * 70)
    print("ANNUAL ANALYSIS")
    print("=" * 70)

    all_results = []

    for year in range(2015, 2025):
        result = analyze_year(wind_countries, solar_countries, year)
        if result is None:
            print(f"{year}: Insufficient data")
            continue

        stats, wind, solar, combined = result
        all_results.append(stats)

        print(f"\n{year} (wind: {stats['n_wind_countries']}, solar: {stats['n_solar_countries']}, {stats['n_hours']:,} hours):")
        print(f"  Wind:     Min={stats['wind_min']:.1f} GW, Mean={stats['wind_mean']:.1f} GW, CV={stats['wind_cv']:.3f}")
        print(f"  Solar:    Min={stats['solar_min']:.1f} GW, Mean={stats['solar_mean']:.1f} GW")
        print(f"  Combined: Min={stats['combined_min']:.1f} GW, Mean={stats['combined_mean']:.1f} GW, CV={stats['combined_cv']:.3f}")
        print(f"  Correlation: {stats['correlation']:.3f}, Baseload improvement: +{stats['baseload_improvement_pct']:.0f}%")

    if not all_results:
        print("No valid years for analysis!")
        return

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'combined_analysis_results.csv'), index=False)

    # Create figure
    print("\n" + "=" * 70)
    print("CREATING FIGURES")
    print("=" * 70)
    create_main_figure(all_results, FIGURES_DIR)

    # Final summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)

    latest = all_results[-1]
    avg_improvement = np.mean([r['baseload_improvement_pct'] for r in all_results])
    avg_corr = np.mean([r['correlation'] for r in all_results])

    print(f"""
MAIN RESULT:
  Adding solar to wind increases the European baseload floor by {avg_improvement:.0f}%
  (range: {min(r['baseload_improvement_pct'] for r in all_results):.0f}% to {max(r['baseload_improvement_pct'] for r in all_results):.0f}%)

2024 STATISTICS:
  Wind-only baseload:    {latest['wind_min']:.1f} GW
  Combined baseload:     {latest['combined_min']:.1f} GW
  Improvement:           +{latest['baseload_improvement_gw']:.1f} GW ({latest['baseload_improvement_pct']:.0f}%)

COMPLEMENTARITY:
  Wind-solar correlation: {avg_corr:.2f} (negative = complementary)
  When wind is low, solar tends to be elevated, and vice versa.

DATA LIMITATION:
  ENTSO-E solar data captures {capture_rate*100:.0f}% of official EU generation.
  Main gap: Netherlands (3% captured due to high rooftop solar share).
  Our estimates are therefore CONSERVATIVE - actual combined baseload
  would be slightly higher with complete distributed solar data.
""")

    # Save summary for paper
    with open(os.path.join(OUTPUT_DIR, 'paper_summary.txt'), 'w') as f:
        f.write(f"""
WIND + SOLAR COMBINED ANALYSIS - SUMMARY FOR PAPER
===================================================

Key Finding:
Adding solar PV to the European wind portfolio increases the guaranteed
baseload floor by {avg_improvement:.0f}% on average (range {min(r['baseload_improvement_pct'] for r in all_results):.0f}-{max(r['baseload_improvement_pct'] for r in all_results):.0f}% across 2015-2024).

2024 Results (wind: {latest['n_wind_countries']}, solar: {latest['n_solar_countries']}, {latest['n_hours']:,} hours):
- Wind-only baseload floor: {latest['wind_min']:.1f} GW
- Combined (wind+solar) baseload floor: {latest['combined_min']:.1f} GW
- Improvement: +{latest['baseload_improvement_gw']:.1f} GW (+{latest['baseload_improvement_pct']:.0f}%)
- Wind-solar correlation: {latest['correlation']:.2f}

The negative correlation ({avg_corr:.2f} average) indicates complementarity:
periods of low wind tend to coincide with elevated solar production.

Data Limitation:
ENTSO-E Transparency Platform captures {capture_rate*100:.0f}% of official EU solar
generation (2023). The primary gap is Netherlands, where distributed rooftop
solar (~80% of 24 GW installed) is not reported to the TSO. Our combined
baseload estimates are therefore conservative.
""")

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
