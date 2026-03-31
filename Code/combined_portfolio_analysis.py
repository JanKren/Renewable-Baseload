# -*- coding: utf-8 -*-
"""
Combined Portfolio Analysis - Three Strengthening Analyses

1. Combined wind+solar Dunkelflaute analysis
2. Country-level wind-solar correlation
3. Seasonal diversification with solar

Authors: Kren & Zajec
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import glob
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

WIND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
SOLAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Solar")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "CombinedAnalysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dunkelflaute thresholds
THRESHOLDS = {
    'severe': 0.10,
    'moderate': 0.20,
    'mild': 0.30
}

MIN_GAP = 6

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

COLORS = {
    'severe': '#b2182b',
    'moderate': '#ef8a62',
    'mild': '#fddbc7',
    'wind': '#2166ac',
    'combined': '#b2182b',
}


def load_country_data(data_dir, source_type='wind'):
    """Load all country data files."""
    if source_type == 'wind':
        pattern = "*_wind_2015_2024.csv"
    else:
        pattern = "*_solar_2015_2024.csv"

    files = glob.glob(os.path.join(data_dir, pattern))
    country_data = {}

    for filepath in files:
        filename = os.path.basename(filepath)
        country_code = filename.split('_')[0]

        try:
            df = pd.read_csv(filepath)
            date_col = df.columns[0]
            df['datetime'] = pd.to_datetime(df[date_col], utc=True)
            df = df.set_index('datetime')
            df.index = df.index.tz_localize(None)

            if source_type == 'wind' and 'Wind Total' in df.columns:
                series = df['Wind Total']
            elif 'Solar' in df.columns:
                series = df['Solar']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                series = df[numeric_cols].fillna(0).sum(axis=1)

            # Resample to hourly
            hourly = series.resample('h').mean()
            country_data[country_code] = hourly

        except Exception as e:
            print(f"  Error loading {country_code} {source_type}: {e}")

    return country_data


def compute_aggregated_production(wind_data, solar_data=None, combined=False):
    """Compute aggregated European production."""

    if combined and solar_data:
        common_countries = set(wind_data.keys()) & set(solar_data.keys())
    else:
        common_countries = set(wind_data.keys())

    # Find common timestamps
    all_indices = []
    for country in common_countries:
        if country in wind_data:
            all_indices.append(set(wind_data[country].dropna().index))

    common_idx = set.intersection(*all_indices) if all_indices else set()
    common_idx = sorted(common_idx)

    # Aggregate
    wind_total = np.zeros(len(common_idx))
    solar_total = np.zeros(len(common_idx))

    for country in common_countries:
        if country in wind_data:
            wind_total += wind_data[country].reindex(common_idx).fillna(0).values
        if combined and solar_data and country in solar_data:
            solar_total += solar_data[country].reindex(common_idx).fillna(0).values

    if combined:
        total = wind_total + solar_total
    else:
        total = wind_total

    return pd.Series(total, index=common_idx), len(common_countries)


def detect_events(series, threshold_value, min_gap=6):
    """Detect Dunkelflaute events below threshold."""
    below = series < threshold_value

    events = []
    in_event = False
    event_start = None
    event_values = []

    for idx, is_below in below.items():
        if is_below and not in_event:
            in_event = True
            event_start = idx
            event_values = [series[idx]]
        elif is_below and in_event:
            event_values.append(series[idx])
        elif not is_below and in_event:
            in_event = False
            events.append({
                'start': event_start,
                'end': idx,
                'duration_hours': len(event_values),
                'min_value': min(event_values),
                'mean_value': np.mean(event_values),
                'deficit': sum(threshold_value - v for v in event_values)
            })
            event_values = []

    if in_event:
        events.append({
            'start': event_start,
            'end': series.index[-1],
            'duration_hours': len(event_values),
            'min_value': min(event_values),
            'mean_value': np.mean(event_values),
            'deficit': sum(threshold_value - v for v in event_values)
        })

    # Merge events with small gaps
    if len(events) > 1:
        merged = [events[0]]
        for event in events[1:]:
            gap = (event['start'] - merged[-1]['end']).total_seconds() / 3600
            if gap <= min_gap:
                merged[-1]['end'] = event['end']
                merged[-1]['duration_hours'] += event['duration_hours'] + int(gap)
                merged[-1]['min_value'] = min(merged[-1]['min_value'], event['min_value'])
                merged[-1]['deficit'] += event['deficit']
            else:
                merged.append(event)
        events = merged

    return events


def analyze_dunkelflaute_comparison(wind_production, combined_production):
    """Analyze Dunkelflaute for wind-only vs combined."""

    results = {'wind': {}, 'combined': {}}

    for name, production in [('wind', wind_production), ('combined', combined_production)]:
        mean_prod = production.mean()

        for severity, threshold_frac in THRESHOLDS.items():
            threshold_value = mean_prod * threshold_frac
            events = detect_events(production, threshold_value, MIN_GAP)

            results[name][severity] = {
                'threshold_gw': threshold_value / 1000,
                'threshold_frac': threshold_frac,
                'events': events,
                'n_events': len(events),
                'total_hours': sum(e['duration_hours'] for e in events),
                'max_duration': max(e['duration_hours'] for e in events) if events else 0,
            }

    return results


def compute_country_wind_solar_correlations(wind_data, solar_data):
    """Compute wind-solar correlation for each country."""

    common_countries = set(wind_data.keys()) & set(solar_data.keys())
    correlations = {}

    for country in common_countries:
        wind = wind_data[country].dropna()
        solar = solar_data[country].dropna()

        # Align timestamps
        common_idx = wind.index.intersection(solar.index)
        # Require at least 5000 hours for reliable correlation
        if len(common_idx) > 5000:
            r = wind.loc[common_idx].corr(solar.loc[common_idx])
            if not np.isnan(r):
                correlations[country] = {
                    'correlation': r,
                    'n_hours': len(common_idx),
                    'wind_mean_gw': wind.loc[common_idx].mean() / 1000,
                    'solar_mean_gw': solar.loc[common_idx].mean() / 1000,
                }

    return correlations


def compute_seasonal_diversification(wind_data, solar_data):
    """Compute seasonal CV reduction for wind-only vs combined."""

    common_countries = set(wind_data.keys()) & set(solar_data.keys())

    # Find common timestamps - use majority approach instead of intersection
    # to avoid losing too much data from countries with gaps
    all_wind_idx = pd.concat([wind_data[c].dropna() for c in common_countries], axis=1)
    all_solar_idx = pd.concat([solar_data[c].dropna() for c in common_countries], axis=1)

    # Keep timestamps where we have at least 20 countries
    wind_counts = all_wind_idx.notna().sum(axis=1)
    solar_counts = all_solar_idx.notna().sum(axis=1)

    valid_wind = wind_counts >= 20
    valid_solar = solar_counts >= 15

    common_idx = valid_wind.index[valid_wind & valid_solar]
    print(f"  Seasonal analysis using {len(common_idx):,} common timestamps")

    # Aggregate
    wind_total = np.zeros(len(common_idx))
    solar_total = np.zeros(len(common_idx))

    for country in common_countries:
        wind_total += wind_data[country].reindex(common_idx).fillna(0).values
        solar_total += solar_data[country].reindex(common_idx).fillna(0).values

    wind_series = pd.Series(wind_total, index=common_idx)
    combined_series = pd.Series(wind_total + solar_total, index=common_idx)

    # Compute CV by season
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11]
    }

    seasonal_stats = []

    for season, months in seasons.items():
        mask = np.array([idx.month in months for idx in common_idx])

        wind_season = wind_series.iloc[mask]
        combined_season = combined_series.iloc[mask]

        # Individual country CVs for this season
        country_wind_cvs = []
        country_combined_cvs = []

        for country in common_countries:
            w = wind_data[country].reindex(common_idx).fillna(0).iloc[mask]
            s = solar_data[country].reindex(common_idx).fillna(0).iloc[mask]

            if w.mean() > 100:  # At least 100 MW average
                country_wind_cvs.append(w.std() / w.mean())
            if (w + s).mean() > 100:
                country_combined_cvs.append((w + s).std() / (w + s).mean())

        wind_cv_agg = wind_season.std() / wind_season.mean() if wind_season.mean() > 0 else np.nan
        combined_cv_agg = combined_season.std() / combined_season.mean() if combined_season.mean() > 0 else np.nan

        wind_cv_ind = np.nanmean(country_wind_cvs) if country_wind_cvs else np.nan
        combined_cv_ind = np.nanmean(country_combined_cvs) if country_combined_cvs else np.nan

        wind_cv_reduction = (wind_cv_ind - wind_cv_agg) / wind_cv_ind * 100 if wind_cv_ind > 0 else np.nan
        combined_cv_reduction = (combined_cv_ind - combined_cv_agg) / combined_cv_ind * 100 if combined_cv_ind > 0 else np.nan

        seasonal_stats.append({
            'season': season,
            'wind_cv_individual': wind_cv_ind,
            'wind_cv_aggregate': wind_cv_agg,
            'wind_cv_reduction': wind_cv_reduction,
            'combined_cv_individual': combined_cv_ind,
            'combined_cv_aggregate': combined_cv_agg,
            'combined_cv_reduction': combined_cv_reduction,
            'wind_mean_gw': wind_season.mean() / 1000,
            'combined_mean_gw': combined_season.mean() / 1000,
            'n_hours': mask.sum(),
        })

    return pd.DataFrame(seasonal_stats)


def create_combined_analysis_figure(dunkelflaute_results, country_correlations,
                                    seasonal_stats, output_dir):
    """Create comprehensive figure for combined portfolio analysis."""

    fig = plt.figure(figsize=(7.08, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.10, right=0.95, top=0.95, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # =========================================================================
    # Panel A: Dunkelflaute comparison - event counts
    # =========================================================================
    severities = ['severe', 'moderate', 'mild']
    x = np.arange(len(severities))
    width = 0.35

    wind_counts = [dunkelflaute_results['wind'][s]['n_events'] for s in severities]
    combined_counts = [dunkelflaute_results['combined'][s]['n_events'] for s in severities]

    ax_a.bar(x - width/2, wind_counts, width, label='Wind only', color=COLORS['wind'])
    ax_a.bar(x + width/2, combined_counts, width, label='Wind + Solar', color=COLORS['combined'])

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(['Severe\n(<10%)', 'Moderate\n(<20%)', 'Mild\n(<30%)'])
    ax_a.set_ylabel('Number of Events')
    ax_a.set_title('a) Dunkelflaute Events (2015-2024)', loc='left', fontweight='bold')
    ax_a.legend(frameon=False, loc='upper right')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Add reduction percentages
    for i, (w, c) in enumerate(zip(wind_counts, combined_counts)):
        if w > 0:
            reduction = (w - c) / w * 100
            ax_a.text(i, max(w, c) + 5, f'-{reduction:.0f}%', ha='center', fontsize=7, color='green')

    # =========================================================================
    # Panel B: Max event duration comparison
    # =========================================================================
    wind_max = [dunkelflaute_results['wind'][s]['max_duration'] for s in severities]
    combined_max = [dunkelflaute_results['combined'][s]['max_duration'] for s in severities]

    ax_b.bar(x - width/2, wind_max, width, label='Wind only', color=COLORS['wind'])
    ax_b.bar(x + width/2, combined_max, width, label='Wind + Solar', color=COLORS['combined'])

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(['Severe', 'Moderate', 'Mild'])
    ax_b.set_ylabel('Max Duration (hours)')
    ax_b.set_title('b) Longest Event Duration', loc='left', fontweight='bold')
    ax_b.legend(frameon=False, loc='upper right')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # =========================================================================
    # Panel C: Country-level wind-solar correlations
    # =========================================================================
    countries = sorted(country_correlations.keys(),
                      key=lambda x: country_correlations[x]['correlation'])
    corrs = [country_correlations[c]['correlation'] for c in countries]

    colors = ['#b2182b' if r < -0.3 else '#ef8a62' if r < -0.1 else
              '#fddbc7' if r < 0.1 else '#d1e5f0' if r < 0.3 else '#2166ac'
              for r in corrs]

    ax_c.barh(range(len(countries)), corrs, color=colors, edgecolor='white', linewidth=0.3)
    ax_c.set_yticks(range(len(countries)))
    ax_c.set_yticklabels(countries, fontsize=5)
    ax_c.set_xlabel('Pearson Correlation')
    ax_c.set_title('c) Wind-Solar Correlation by Country', loc='left', fontweight='bold')
    ax_c.axvline(0, color='black', linewidth=0.5)
    ax_c.set_xlim(-0.6, 0.3)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Add mean line
    mean_corr = np.mean(corrs)
    ax_c.axvline(mean_corr, color='red', linestyle='--', linewidth=1)
    ax_c.text(mean_corr + 0.02, len(countries) - 2, f'Mean: {mean_corr:.2f}',
              fontsize=7, color='red')

    # =========================================================================
    # Panel D: Seasonal CV reduction comparison
    # =========================================================================
    seasons = seasonal_stats['season'].values
    x = np.arange(len(seasons))

    wind_red = seasonal_stats['wind_cv_reduction'].values
    combined_red = seasonal_stats['combined_cv_reduction'].values

    ax_d.bar(x - width/2, wind_red, width, label='Wind only', color=COLORS['wind'])
    ax_d.bar(x + width/2, combined_red, width, label='Wind + Solar', color=COLORS['combined'])

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(seasons)
    ax_d.set_ylabel('CV Reduction (%)')
    ax_d.set_title('d) Seasonal Diversification Benefit', loc='left', fontweight='bold')
    ax_d.legend(frameon=False, loc='lower right')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # Add improvement annotation
    for i in range(len(seasons)):
        improvement = combined_red[i] - wind_red[i]
        if improvement > 0:
            ax_d.text(i + width/2, combined_red[i] + 1, f'+{improvement:.0f}%',
                     ha='center', fontsize=6, color='green')

    # =========================================================================
    # Panel E: Summary statistics table
    # =========================================================================
    ax_e.axis('off')

    # Create summary table
    table_data = [
        ['Metric', 'Wind Only', 'Wind + Solar', 'Improvement'],
        ['Severe events (<10%)',
         str(dunkelflaute_results['wind']['severe']['n_events']),
         str(dunkelflaute_results['combined']['severe']['n_events']),
         f"{dunkelflaute_results['wind']['severe']['n_events'] - dunkelflaute_results['combined']['severe']['n_events']} fewer"],
        ['Moderate events (<20%)',
         str(dunkelflaute_results['wind']['moderate']['n_events']),
         str(dunkelflaute_results['combined']['moderate']['n_events']),
         f"-{(dunkelflaute_results['wind']['moderate']['n_events'] - dunkelflaute_results['combined']['moderate']['n_events']) / max(1, dunkelflaute_results['wind']['moderate']['n_events']) * 100:.0f}%"],
        ['Max moderate duration (h)',
         str(dunkelflaute_results['wind']['moderate']['max_duration']),
         str(dunkelflaute_results['combined']['moderate']['max_duration']),
         f"-{(dunkelflaute_results['wind']['moderate']['max_duration'] - dunkelflaute_results['combined']['moderate']['max_duration']) / max(1, dunkelflaute_results['wind']['moderate']['max_duration']) * 100:.0f}%"],
        ['Mean wind-solar corr.', '-', f"{mean_corr:.3f}", 'Complementary'],
        ['Winter CV reduction', f"{seasonal_stats[seasonal_stats['season']=='Winter']['wind_cv_reduction'].values[0]:.1f}%",
         f"{seasonal_stats[seasonal_stats['season']=='Winter']['combined_cv_reduction'].values[0]:.1f}%",
         f"+{seasonal_stats[seasonal_stats['season']=='Winter']['combined_cv_reduction'].values[0] - seasonal_stats[seasonal_stats['season']=='Winter']['wind_cv_reduction'].values[0]:.1f}%"],
        ['Summer CV reduction', f"{seasonal_stats[seasonal_stats['season']=='Summer']['wind_cv_reduction'].values[0]:.1f}%",
         f"{seasonal_stats[seasonal_stats['season']=='Summer']['combined_cv_reduction'].values[0]:.1f}%",
         f"+{seasonal_stats[seasonal_stats['season']=='Summer']['combined_cv_reduction'].values[0] - seasonal_stats[seasonal_stats['season']=='Summer']['wind_cv_reduction'].values[0]:.1f}%"],
    ]

    table = ax_e.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#d9d9d9')
        table[(0, i)].set_text_props(fontweight='bold')

    ax_e.set_title('e) Summary: Combined Portfolio Benefits', loc='left', fontweight='bold', y=0.95)

    # Save
    output_path = os.path.join(output_dir, 'figure_combined_portfolio_analysis.pdf')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("COMBINED PORTFOLIO ANALYSIS")
    print("1. Wind+Solar Dunkelflaute")
    print("2. Country Wind-Solar Correlations")
    print("3. Seasonal Diversification with Solar")
    print("=" * 70)

    # Load data
    print("\nLoading wind data...")
    wind_data = load_country_data(WIND_DIR, 'wind')
    print(f"  Loaded {len(wind_data)} countries")

    print("\nLoading solar data...")
    solar_data = load_country_data(SOLAR_DIR, 'solar')
    print(f"  Loaded {len(solar_data)} countries")

    common = set(wind_data.keys()) & set(solar_data.keys())
    print(f"\nCountries with both wind and solar: {len(common)}")
    print(f"  {sorted(common)}")

    # =========================================================================
    # Analysis 1: Dunkelflaute comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: DUNKELFLAUTE COMPARISON")
    print("=" * 70)

    print("\nComputing wind-only aggregated production...")
    wind_production, n_wind = compute_aggregated_production(wind_data, combined=False)
    print(f"  {len(wind_production):,} hours, {n_wind} countries")
    print(f"  Mean: {wind_production.mean()/1000:.1f} GW, Min: {wind_production.min()/1000:.1f} GW")

    print("\nComputing combined wind+solar aggregated production...")
    combined_production, n_combined = compute_aggregated_production(wind_data, solar_data, combined=True)
    print(f"  {len(combined_production):,} hours, {n_combined} countries")
    print(f"  Mean: {combined_production.mean()/1000:.1f} GW, Min: {combined_production.min()/1000:.1f} GW")

    print("\nAnalyzing Dunkelflaute events...")
    dunkelflaute_results = analyze_dunkelflaute_comparison(wind_production, combined_production)

    print("\nWIND ONLY:")
    for sev in ['severe', 'moderate', 'mild']:
        r = dunkelflaute_results['wind'][sev]
        print(f"  {sev.capitalize()}: {r['n_events']} events, max {r['max_duration']}h, total {r['total_hours']}h")

    print("\nCOMBINED WIND+SOLAR:")
    for sev in ['severe', 'moderate', 'mild']:
        r = dunkelflaute_results['combined'][sev]
        print(f"  {sev.capitalize()}: {r['n_events']} events, max {r['max_duration']}h, total {r['total_hours']}h")

    # =========================================================================
    # Analysis 2: Country-level wind-solar correlations
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: COUNTRY WIND-SOLAR CORRELATIONS")
    print("=" * 70)

    country_correlations = compute_country_wind_solar_correlations(wind_data, solar_data)

    # Sort by correlation
    sorted_countries = sorted(country_correlations.items(), key=lambda x: x[1]['correlation'])

    print("\nCountry wind-solar correlations (sorted):")
    for country, data in sorted_countries:
        print(f"  {country}: r = {data['correlation']:.3f} (n = {data['n_hours']:,})")

    corr_values = [d['correlation'] for d in country_correlations.values() if not np.isnan(d['correlation'])]
    mean_corr = np.mean(corr_values)
    print(f"\nMean correlation: {mean_corr:.3f}")
    print(f"Most negative: {sorted_countries[0][0]} (r = {sorted_countries[0][1]['correlation']:.3f})")
    print(f"Most positive: {sorted_countries[-1][0]} (r = {sorted_countries[-1][1]['correlation']:.3f})")

    # =========================================================================
    # Analysis 3: Seasonal diversification
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: SEASONAL DIVERSIFICATION")
    print("=" * 70)

    seasonal_stats = compute_seasonal_diversification(wind_data, solar_data)

    print("\nSeasonal CV reduction (wind vs combined):")
    for _, row in seasonal_stats.iterrows():
        print(f"  {row['season']}: Wind {row['wind_cv_reduction']:.1f}% -> Combined {row['combined_cv_reduction']:.1f}% "
              f"(+{row['combined_cv_reduction'] - row['wind_cv_reduction']:.1f}%)")

    # =========================================================================
    # Create figure
    # =========================================================================
    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)

    output_path = create_combined_analysis_figure(
        dunkelflaute_results, country_correlations, seasonal_stats, OUTPUT_DIR)

    # =========================================================================
    # Save results
    # =========================================================================
    print("\nSaving results...")

    # Dunkelflaute comparison
    dunk_summary = []
    for source in ['wind', 'combined']:
        for sev in ['severe', 'moderate', 'mild']:
            r = dunkelflaute_results[source][sev]
            dunk_summary.append({
                'source': source,
                'severity': sev,
                'threshold_gw': r['threshold_gw'],
                'n_events': r['n_events'],
                'total_hours': r['total_hours'],
                'max_duration_hours': r['max_duration'],
            })
    pd.DataFrame(dunk_summary).to_csv(
        os.path.join(RESULTS_DIR, 'dunkelflaute_comparison.csv'), index=False)

    # Country correlations
    corr_df = pd.DataFrame([
        {'country': c, **d} for c, d in country_correlations.items()
    ]).sort_values('correlation')
    corr_df.to_csv(os.path.join(RESULTS_DIR, 'country_wind_solar_correlations.csv'), index=False)

    # Seasonal stats
    seasonal_stats.to_csv(os.path.join(RESULTS_DIR, 'seasonal_diversification.csv'), index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    # Print key findings for paper
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)

    wind_mod = dunkelflaute_results['wind']['moderate']
    comb_mod = dunkelflaute_results['combined']['moderate']

    print(f"\n1. COMBINED DUNKELFLAUTE:")
    print(f"   - Moderate events: {wind_mod['n_events']} (wind) -> {comb_mod['n_events']} (combined)")
    print(f"   - Reduction: {(wind_mod['n_events'] - comb_mod['n_events']) / wind_mod['n_events'] * 100:.0f}%")
    print(f"   - Max duration: {wind_mod['max_duration']}h (wind) -> {comb_mod['max_duration']}h (combined)")

    print(f"\n2. COUNTRY CORRELATIONS:")
    print(f"   - Mean: r = {mean_corr:.3f}")
    print(f"   - Range: {sorted_countries[0][1]['correlation']:.3f} to {sorted_countries[-1][1]['correlation']:.3f}")
    print(f"   - {sum(1 for c, d in country_correlations.items() if d['correlation'] < 0)} of {len(country_correlations)} countries show negative correlation")

    winter = seasonal_stats[seasonal_stats['season'] == 'Winter'].iloc[0]
    summer = seasonal_stats[seasonal_stats['season'] == 'Summer'].iloc[0]

    print(f"\n3. SEASONAL DIVERSIFICATION:")
    print(f"   - Winter: {winter['wind_cv_reduction']:.1f}% (wind) -> {winter['combined_cv_reduction']:.1f}% (combined)")
    print(f"   - Summer: {summer['wind_cv_reduction']:.1f}% (wind) -> {summer['combined_cv_reduction']:.1f}% (combined)")
    print(f"   - Winter improvement: +{winter['combined_cv_reduction'] - winter['wind_cv_reduction']:.1f}%")


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
