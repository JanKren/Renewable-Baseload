# -*- coding: utf-8 -*-
"""
Dunkelflaute Analysis - Kittel & Schill (2026) Framework

Characterizes low-wind events (Dunkelflaute) in European wind production
using actual production data from 2015-2024.

Multi-threshold framework:
- Level 1 (Severe): < 10% of mean production
- Level 2 (Moderate): < 20% of mean production
- Level 3 (Mild): < 30% of mean production

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
SOLAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Solar")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Dunkelflaute")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Kittel & Schill (2026) thresholds (as fraction of mean)
THRESHOLDS = {
    'severe': 0.10,    # < 10% of mean
    'moderate': 0.20,  # < 20% of mean
    'mild': 0.30       # < 30% of mean
}

# Minimum gap to consider separate events (hours)
MIN_GAP = 6

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
    'severe': '#b2182b',    # Dark red
    'moderate': '#ef8a62',  # Light red
    'mild': '#fddbc7',      # Pale orange
    'normal': '#2166ac',    # Blue
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


def compute_aggregated_production(country_data):
    """Compute hourly aggregated European production."""
    all_series = []

    for country, df in country_data.items():
        # Resample to hourly
        hourly = df['wind'].resample('h').mean()
        all_series.append(hourly)

    # Combine all series
    combined = pd.concat(all_series, axis=1)
    combined.columns = list(country_data.keys())

    # Sum across countries (ignoring NaN)
    total = combined.sum(axis=1, skipna=True)

    # Remove rows where we have too few countries
    valid_counts = combined.notna().sum(axis=1)
    total = total[valid_counts >= 20]  # Require at least 20 countries

    return total


def load_all_solar_data(data_dir):
    """Load all country solar data files."""
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
            df = df.drop(columns=[date_col], errors='ignore')

            if 'Solar' in df.columns:
                df['solar'] = df['Solar']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df['solar'] = df[numeric_cols].fillna(0).sum(axis=1)

            country_data[country_code] = df[['solar']]

        except Exception as e:
            print(f"  Error loading solar {country_code}: {e}")

    return country_data


def compute_aggregated_solar(country_data):
    """Compute hourly aggregated European solar production."""
    all_series = []

    for country, df in country_data.items():
        hourly = df['solar'].resample('h').mean()
        all_series.append(hourly)

    combined = pd.concat(all_series, axis=1)
    combined.columns = list(country_data.keys())

    total = combined.sum(axis=1, skipna=True)

    valid_counts = combined.notna().sum(axis=1)
    total = total[valid_counts >= 10]  # Require at least 10 solar countries

    return total


def detect_events(series, threshold_value, min_gap=6):
    """
    Detect Dunkelflaute events below threshold.

    Parameters:
    - series: Hourly production time series
    - threshold_value: Absolute threshold (MW)
    - min_gap: Minimum hours between events to consider separate

    Returns:
    - List of event dictionaries with start, end, duration, min_value, deficit
    """
    below = series < threshold_value

    events = []
    in_event = False
    event_start = None
    event_values = []

    for idx, is_below in below.items():
        if is_below and not in_event:
            # Start new event
            in_event = True
            event_start = idx
            event_values = [series[idx]]
        elif is_below and in_event:
            # Continue event
            event_values.append(series[idx])
        elif not is_below and in_event:
            # End event
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

    # Handle event at end of series
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
                # Merge with previous
                merged[-1]['end'] = event['end']
                merged[-1]['duration_hours'] += event['duration_hours'] + int(gap)
                merged[-1]['min_value'] = min(merged[-1]['min_value'], event['min_value'])
                merged[-1]['deficit'] += event['deficit']
            else:
                merged.append(event)
        events = merged

    return events


def analyze_dunkelflaute(total_production):
    """Analyze Dunkelflaute events using FIXED decade-mean thresholds (for comparison only)."""

    mean_production = total_production.mean()
    print(f"\nMean European wind production (decade): {mean_production/1000:.1f} GW")

    results = {}

    for severity, threshold_frac in THRESHOLDS.items():
        threshold_value = mean_production * threshold_frac
        print(f"\n[FIXED] {severity.upper()} threshold: < {threshold_value/1000:.1f} GW ({threshold_frac*100:.0f}% of decade mean)")

        events = detect_events(total_production, threshold_value, MIN_GAP)

        results[severity] = {
            'threshold_gw': threshold_value / 1000,
            'threshold_frac': threshold_frac,
            'events': events,
            'n_events': len(events),
            'total_hours': sum(e['duration_hours'] for e in events),
            'max_duration': max(e['duration_hours'] for e in events) if events else 0,
            'total_deficit_twh': sum(e['deficit'] for e in events) / 1e6 if events else 0
        }

        print(f"  Number of events: {len(events)}")
        print(f"  Total hours below threshold: {results[severity]['total_hours']}")

    return results


def analyze_dunkelflaute_yearly(total_production):
    """Analyze Dunkelflaute events using YEAR-NORMALIZED thresholds (primary analysis).

    Each year's thresholds are defined relative to that year's own mean production,
    isolating meteorological variability from the confounding effect of capacity growth.
    """
    years = range(2015, 2025)
    results = {}

    for severity, threshold_frac in THRESHOLDS.items():
        all_events = []
        for year in years:
            yr = total_production[f'{year}']
            yr_mean = yr.mean()
            threshold_value = yr_mean * threshold_frac
            events = detect_events(yr, threshold_value, MIN_GAP)
            all_events.extend(events)

        results[severity] = {
            'threshold_frac': threshold_frac,
            'events': all_events,
            'n_events': len(all_events),
            'total_hours': sum(e['duration_hours'] for e in all_events),
            'max_duration': max(e['duration_hours'] for e in all_events) if all_events else 0,
            'total_deficit_twh': sum(e['deficit'] for e in all_events) / 1e6 if all_events else 0
        }

        print(f"\n[YEAR-NORM] {severity.upper()} ({threshold_frac*100:.0f}% of year mean): "
              f"{len(all_events)} events, {results[severity]['total_hours']} hours, "
              f"max duration {results[severity]['max_duration']}h")

        if all_events:
            sorted_events = sorted(all_events, key=lambda x: x['duration_hours'], reverse=True)[:5]
            print(f"  Top 5 longest events:")
            for i, e in enumerate(sorted_events, 1):
                print(f"    {i}. {e['start'].strftime('%Y-%m-%d')} to {e['end'].strftime('%Y-%m-%d')}: "
                      f"{e['duration_hours']} hours ({e['duration_hours']/24:.1f} days), "
                      f"min={e['min_value']/1000:.1f} GW")

    return results


def analyze_seasonal_distribution(results, total_production):
    """Analyze seasonal distribution of events."""

    seasonal_data = defaultdict(lambda: defaultdict(int))

    for severity, data in results.items():
        for event in data['events']:
            month = event['start'].month
            seasonal_data[severity][month] += event['duration_hours']

    return seasonal_data


def compute_yearly_normalized_hours(total_production, combined_production=None):
    """Compute moderate event hours per year using year-specific thresholds."""
    years = range(2015, 2025)
    fixed_hours = []
    normalized_hours = []
    combined_hours = []
    yearly_means = []
    yearly_mins = []
    decade_mean = total_production.mean()
    fixed_threshold = 0.20 * decade_mean

    for year in years:
        yr = total_production[f'{year}']
        yr_mean = yr.mean()
        yr_threshold = 0.20 * yr_mean
        yearly_means.append(yr_mean / 1000)
        yearly_mins.append(yr.min() / 1000)
        fixed_hours.append(len(yr[yr < fixed_threshold]))
        normalized_hours.append(len(yr[yr < yr_threshold]))

        if combined_production is not None:
            c_yr = combined_production[f'{year}']
            c_mean = c_yr.mean()
            c_threshold = 0.20 * c_mean
            combined_hours.append(len(c_yr[c_yr < c_threshold]))
        else:
            combined_hours.append(0)

    return {
        'years': list(years),
        'fixed_hours': fixed_hours,
        'normalized_hours': normalized_hours,
        'combined_hours': combined_hours,
        'yearly_means': yearly_means,
        'yearly_mins': yearly_mins,
        'fixed_threshold_gw': fixed_threshold / 1000,
    }


def create_dunkelflaute_figure(total_production, results_yearnorm, seasonal_data_yearnorm,
                               yearly_data, output_dir):
    """Create Figure 3: Dunkelflaute characterization using year-normalized thresholds."""

    fig = plt.figure(figsize=(7.08, 5.5))  # 180mm wide, more compact

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.65], width_ratios=[1.3, 1],
                          hspace=0.45, wspace=0.30,
                          left=0.08, right=0.96, top=0.96, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])  # Duration distribution
    ax_b = fig.add_subplot(gs[0, 1])  # Seasonal heatmap
    ax_c = fig.add_subplot(gs[1, :])  # Year-normalized vs fixed comparison

    # =========================================================================
    # Panel A: Event Duration Distribution (YEAR-NORMALIZED thresholds)
    # =========================================================================
    for severity in ['mild', 'moderate', 'severe']:
        events = results_yearnorm[severity]['events']
        if events:
            durations = [e['duration_hours'] for e in events]
            bins = np.logspace(0, np.log10(max(durations) + 1), 20)
            ax_a.hist(durations, bins=bins, alpha=0.7, label=f"{severity.capitalize()}",
                     color=COLORS[severity], edgecolor='white', linewidth=0.5)

    ax_a.set_xscale('log')
    ax_a.set_xlabel('Event Duration [hours]')
    ax_a.set_ylabel('Number of Events')
    ax_a.set_title('a) Duration distribution (year-normalized)', loc='left', fontweight='bold')
    ax_a.legend(frameon=False, loc='upper left')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Add day markers
    for days in [1, 7, 30]:
        ax_a.axvline(days * 24, color='gray', linestyle=':', alpha=0.5)
        ax_a.text(days * 24, ax_a.get_ylim()[1] * 0.95, f'{days}d', fontsize=6,
                 ha='center', color='gray')

    # =========================================================================
    # Panel B: Seasonal Distribution Heatmap (YEAR-NORMALIZED thresholds)
    # =========================================================================
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    heatmap_data = np.zeros((3, 12))
    for i, severity in enumerate(['severe', 'moderate', 'mild']):
        for month in range(1, 13):
            heatmap_data[i, month-1] = seasonal_data_yearnorm[severity].get(month, 0)

    # Normalize by hours per year
    heatmap_data = heatmap_data / 10  # 10 years, show annual average hours

    sns.heatmap(heatmap_data, ax=ax_b, cmap='YlOrRd',
                xticklabels=months, yticklabels=['Severe', 'Moderate', 'Mild'],
                annot=True, fmt='.0f', annot_kws={'fontsize': 6},
                cbar_kws={'label': 'Hours/year', 'shrink': 0.8},
                linewidths=0.5, linecolor='white')

    ax_b.set_title('b) Seasonal pattern (year-normalized)', loc='left', fontweight='bold')
    ax_b.set_xlabel('')
    ax_b.set_ylabel('')

    # =========================================================================
    # Panel C: Year-Normalized Moderate Event Hours (Wind-only vs Combined)
    # =========================================================================
    years = yearly_data['years']
    x = np.arange(len(years))
    has_combined = any(h > 0 for h in yearly_data['combined_hours'])

    if has_combined:
        width = 0.35
        bars_wind = ax_c.bar(x - width/2, yearly_data['normalized_hours'], width,
                             color=COLORS['normal'], alpha=0.85,
                             label='Wind only',
                             edgecolor='white', linewidth=0.5)
        bars_comb = ax_c.bar(x + width/2, yearly_data['combined_hours'], width,
                             color=COLORS['severe'], alpha=0.85,
                             label='Wind + solar',
                             edgecolor='white', linewidth=0.5)
        # Value labels
        for bar in bars_wind:
            h = bar.get_height()
            if h > 0:
                ax_c.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{int(h)}',
                         ha='center', va='bottom', fontsize=5.5, color=COLORS['normal'])
        for bar in bars_comb:
            h = bar.get_height()
            if h > 0:
                ax_c.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{int(h)}',
                         ha='center', va='bottom', fontsize=5.5, color=COLORS['severe'])
    else:
        width = 0.55
        bars_wind = ax_c.bar(x, yearly_data['normalized_hours'], width,
                             color=COLORS['normal'], alpha=0.85,
                             label='Moderate hours (20% of annual mean)',
                             edgecolor='white', linewidth=0.5)
        for bar in bars_wind:
            h = bar.get_height()
            if h > 0:
                ax_c.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{int(h)}',
                         ha='center', va='bottom', fontsize=6, color=COLORS['normal'])

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(years)
    ax_c.set_xlabel('Year')
    ax_c.set_ylabel('Hours below threshold')
    ax_c.set_title('c) Moderate drought hours per year (year-normalized)',
                   loc='left', fontweight='bold')
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Add secondary axis showing annual mean production
    ax_c2 = ax_c.twinx()
    ax_c2.plot(x, yearly_data['yearly_means'], 'k--', alpha=0.4, linewidth=1,
               label='Annual mean wind production')
    ax_c2.set_ylabel('Mean wind production [GW]', color='gray')
    ax_c2.tick_params(axis='y', labelcolor='gray')
    ax_c2.spines['top'].set_visible(False)

    # Combine legends from both axes, place at upper left corner
    handles1, labels1 = ax_c.get_legend_handles_labels()
    handles2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(handles1 + handles2, labels1 + labels2,
                frameon=False, fontsize=6, loc='upper left',
                ncol=1)

    # Save figure
    output_path = os.path.join(output_dir, 'figure3_dunkelflaute.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")


def save_results(results, output_dir):
    """Save Dunkelflaute analysis results to CSV."""

    # Summary statistics
    summary = []
    for severity, data in results.items():
        row = {
            'severity': severity,
            'threshold_fraction': data['threshold_frac'],
            'n_events': data['n_events'],
            'total_hours': data['total_hours'],
            'max_duration_hours': data['max_duration'],
            'max_duration_days': data['max_duration'] / 24,
            'total_deficit_twh': data['total_deficit_twh']
        }
        if 'threshold_gw' in data:
            row['threshold_gw'] = data['threshold_gw']
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_dir, 'dunkelflaute_summary.csv'), index=False)
    print(f"Saved: {output_dir}/dunkelflaute_summary.csv")

    # All events
    all_events = []
    for severity, data in results.items():
        for event in data['events']:
            all_events.append({
                'severity': severity,
                'start': event['start'],
                'end': event['end'],
                'duration_hours': event['duration_hours'],
                'duration_days': event['duration_hours'] / 24,
                'min_value_gw': event['min_value'] / 1000,
                'mean_value_gw': event['mean_value'] / 1000,
                'deficit_twh': event['deficit'] / 1e6
            })

    events_df = pd.DataFrame(all_events)
    events_df = events_df.sort_values(['severity', 'duration_hours'], ascending=[True, False])
    events_df.to_csv(os.path.join(output_dir, 'dunkelflaute_events.csv'), index=False)
    print(f"Saved: {output_dir}/dunkelflaute_events.csv")


def main():
    print("=" * 70)
    print("DUNKELFLAUTE ANALYSIS (Kittel & Schill 2026 Framework)")
    print("=" * 70)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    print("\nComputing aggregated European production...")
    total_production = compute_aggregated_production(country_data)
    print(f"Total observations: {len(total_production):,} hours")
    print(f"Date range: {total_production.index.min()} to {total_production.index.max()}")

    # Primary analysis: year-normalized thresholds (used for figure panels A & B)
    print("\n--- PRIMARY: Year-normalized thresholds ---")
    results_yearnorm = analyze_dunkelflaute_yearly(total_production)
    seasonal_data_yearnorm = analyze_seasonal_distribution(results_yearnorm, total_production)

    # Secondary: fixed decade-mean thresholds (for comparison in panel C and CSV)
    print("\n--- SECONDARY: Fixed decade-mean thresholds ---")
    results_fixed = analyze_dunkelflaute(total_production)

    # Load solar data for combined wind+solar drought analysis
    print("\nLoading solar data...")
    solar_data = load_all_solar_data(SOLAR_DIR)
    print(f"Loaded {len(solar_data)} solar countries")
    aggregated_solar = compute_aggregated_solar(solar_data)
    print(f"Solar observations: {len(aggregated_solar):,} hours")

    # Compute combined wind+solar on common timestamps
    common_idx = total_production.index.intersection(aggregated_solar.index)
    combined_production = total_production.loc[common_idx] + aggregated_solar.loc[common_idx]
    print(f"Combined wind+solar: {len(combined_production):,} hours on common timestamps")

    print("\nComputing yearly comparison data...")
    yearly_data = compute_yearly_normalized_hours(total_production, combined_production)
    for i, year in enumerate(yearly_data['years']):
        print(f"  {year}: mean={yearly_data['yearly_means'][i]:.1f} GW, "
              f"wind-only={yearly_data['normalized_hours'][i]}h, "
              f"combined={yearly_data['combined_hours'][i]}h, "
              f"fixed={yearly_data['fixed_hours'][i]}h")

    print("\nCreating figure (year-normalized thresholds for panels a,b)...")
    create_dunkelflaute_figure(total_production, results_yearnorm,
                               seasonal_data_yearnorm, yearly_data, OUTPUT_DIR)

    print("\nSaving results (year-normalized)...")
    save_results(results_yearnorm, RESULTS_DIR)

    print("\n" + "=" * 70)
    print("DUNKELFLAUTE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
