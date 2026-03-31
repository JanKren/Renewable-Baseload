#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battery Storage Analysis - Combined Wind + Solar

Compares battery requirements for wind-only vs combined wind+solar portfolios.
The combined portfolio has a higher baseline, requiring less storage for
the same target floor level.

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
WIND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
SOLAR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Solar")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Battery")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Target floor levels (GW) - now starting higher since combined baseline is ~17 GW
TARGET_FLOORS_WIND = [15, 20, 25, 30, 35, 40]
TARGET_FLOORS_COMBINED = [20, 25, 30, 35, 40, 45]

# Battery cost assumptions
BATTERY_COST_PER_KWH = 150  # $/kWh
BATTERY_COST_PER_KW = 200   # $/kW

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'figure.dpi': 300,
})


def load_wind_data(data_dir):
    """Load wind data for all countries."""
    country_data = {}
    files = glob.glob(os.path.join(data_dir, "*_wind_2015_2024.csv"))

    for filepath in files:
        country_code = os.path.basename(filepath).split('_')[0]
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)
            df = df.set_index('datetime')
            df.index = df.index.tz_localize(None)

            if 'Wind Total' in df.columns:
                series = df['Wind Total']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                series = df[numeric_cols].fillna(0).sum(axis=1)

            country_data[country_code] = series.resample('h').mean()
        except Exception as e:
            print(f"  Error loading wind {country_code}: {e}")

    return country_data


def load_solar_data(data_dir):
    """Load solar data for all countries."""
    country_data = {}
    files = glob.glob(os.path.join(data_dir, "*_solar_2015_2024.csv"))

    for filepath in files:
        country_code = os.path.basename(filepath).split('_')[0]
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df.iloc[:, 0], utc=True)
            df = df.set_index('datetime')
            df.index = df.index.tz_localize(None)

            if 'Solar' in df.columns:
                series = df['Solar']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                series = df[numeric_cols].fillna(0).sum(axis=1)

            country_data[country_code] = series.resample('h').mean()
        except Exception as e:
            print(f"  Error loading solar {country_code}: {e}")

    return country_data


def get_aggregated_timeseries(wind_data, solar_data=None):
    """Get aggregated European timeseries (wind-only or combined).

    Only includes countries with >8000 hours per year (near-complete coverage)
    to avoid the intersection problem that drops most of the year.
    """
    all_years = []

    for year in range(2015, 2025):
        # Select countries with near-complete data for this year
        wind_year = {}
        for country in wind_data:
            w = wind_data[country][wind_data[country].index.year == year].dropna()
            if len(w) > 8000:
                wind_year[country] = w

        solar_year = {}
        if solar_data:
            for country in solar_data:
                s = solar_data[country][solar_data[country].index.year == year].dropna()
                if len(s) > 8000:
                    solar_year[country] = s

        if len(wind_year) < 3:
            continue

        # Find common timestamps across well-covered countries
        all_indices = [set(s.index) for s in wind_year.values()]
        if solar_data:
            all_indices += [set(s.index) for s in solar_year.values() if len(s) > 0]

        common_idx = set.intersection(*all_indices) if all_indices else set()
        common_idx = sorted(common_idx)

        if len(common_idx) < 100:
            continue

        # Aggregate
        total = pd.Series(0.0, index=common_idx)
        for s in wind_year.values():
            total += s.reindex(common_idx).fillna(0)

        if solar_data:
            for s in solar_year.values():
                total += s.reindex(common_idx).fillna(0)

        all_years.append(total)

    if not all_years:
        return pd.Series(dtype=float)

    return pd.concat(all_years) / 1000  # Convert to GW


def analyze_storage(ts_gw, target_floor):
    """Analyze storage requirements for a target floor."""
    deficit = np.maximum(0, target_floor - ts_gw.values)
    in_deficit = deficit > 0

    events = []
    i = 0
    while i < len(deficit):
        if in_deficit[i]:
            start_idx = i
            event_deficit = []
            while i < len(deficit) and in_deficit[i]:
                event_deficit.append(deficit[i])
                i += 1
            events.append({
                'duration_hours': len(event_deficit),
                'max_power_gw': max(event_deficit),
                'energy_gwh': sum(event_deficit),
            })
        else:
            i += 1

    max_power = np.max(deficit) if len(deficit) > 0 else 0
    max_energy = max([e['energy_gwh'] for e in events]) if events else 0
    max_duration = max([e['duration_hours'] for e in events]) if events else 0

    return {
        'target_floor_gw': target_floor,
        'max_power_gw': max_power,
        'max_energy_gwh': max_energy,
        'n_events': len(events),
        'max_event_duration': max_duration,
        'events': events,
    }


def compute_cost(result):
    """Compute battery cost in billions USD."""
    cost_energy = result['max_energy_gwh'] * 1e6 * BATTERY_COST_PER_KWH / 1e9
    cost_power = result['max_power_gw'] * 1e6 * BATTERY_COST_PER_KW / 1e9
    return cost_energy + cost_power


def create_comparison_figure(wind_ts, combined_ts, wind_results, combined_results, output_dir):
    """Create comparison figure for wind-only vs combined battery requirements."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Baseload comparison
    ax = axes[0]
    wind_2024 = wind_ts[wind_ts.index.year == 2024]
    combined_2024 = combined_ts[combined_ts.index.year == 2024]

    # Sort for duration curve - use hours for x-axis
    wind_sorted = np.sort(wind_2024.values)
    combined_sorted = np.sort(combined_2024.values)

    wind_hours = np.arange(len(wind_sorted))
    combined_hours = np.arange(len(combined_sorted))

    ax.plot(wind_hours, wind_sorted, color='#2166ac', linewidth=1.5, label='Wind only')
    ax.plot(combined_hours, combined_sorted, color='#b2182b', linewidth=1.5, label='Wind + Solar')
    ax.axhline(wind_sorted[0], color='#2166ac', linestyle='--', alpha=0.7)
    ax.axhline(combined_sorted[0], color='#b2182b', linestyle='--', alpha=0.7)
    ax.set_xlabel('Hours')
    ax.set_ylabel('Production [GW]')
    ax.set_title('a) Duration curve (2024)', loc='left', fontweight='bold')
    ax.legend(loc='lower right', fontsize=7)
    ax.text(len(wind_hours)*0.05, wind_sorted[0] + 1, f'Min: {wind_sorted[0]:.1f} GW', color='#2166ac', fontsize=7)
    ax.text(len(combined_hours)*0.05, combined_sorted[0] + 3, f'Min: {combined_sorted[0]:.1f} GW', color='#b2182b', fontsize=7)

    # Panel B: Storage requirements + cost (merged, twin y-axis)
    ax = axes[1]

    # Find common target floors
    common_floors = [20, 25, 30, 35, 40]

    wind_energy = [next((r['max_energy_gwh'] for r in wind_results if r['target_floor_gw'] == f), 0) for f in common_floors]
    combined_energy = [next((r['max_energy_gwh'] for r in combined_results if r['target_floor_gw'] == f), 0) for f in common_floors]

    wind_cost = [compute_cost(next((r for r in wind_results if r['target_floor_gw'] == f), {'max_energy_gwh': 0, 'max_power_gw': 0})) for f in common_floors]
    combined_cost = [compute_cost(next((r for r in combined_results if r['target_floor_gw'] == f), {'max_energy_gwh': 0, 'max_power_gw': 0})) for f in common_floors]

    x = np.arange(len(common_floors))
    width = 0.35

    ax.bar(x - width/2, wind_energy, width, label='Wind only (GWh)', color='#2166ac')
    ax.bar(x + width/2, combined_energy, width, label='Wind + Solar (GWh)', color='#b2182b')
    ax.set_xticks(x)
    ax.set_xticklabels(common_floors)
    ax.set_xlabel('Target Floor [GW]')
    ax.set_ylabel('Required Energy Storage [GWh]')
    ax.set_title('b) Storage requirements and cost', loc='left', fontweight='bold')
    ax.legend(loc='upper left', fontsize=6)

    # Twin y-axis for cost
    ax2_twin = ax.twinx()
    ax2_twin.plot(x - width/2, wind_cost, 's-', color='#2166ac', markersize=5, linewidth=1.5, alpha=0.7)
    ax2_twin.plot(x + width/2, combined_cost, 's-', color='#b2182b', markersize=5, linewidth=1.5, alpha=0.7)
    ax2_twin.set_ylabel('Estimated Cost [$B]')
    ax2_twin.set_ylim(bottom=0)

    # Panel C: Savings from adding solar
    ax = axes[2]

    savings = [(w - c) for w, c in zip(wind_cost, combined_cost)]
    savings_pct = [(w - c) / w * 100 if w > 0 else 0 for w, c in zip(wind_cost, combined_cost)]

    bars = ax.bar(x, savings, color='#4daf4a')
    ax.set_xticks(x)
    ax.set_xticklabels(common_floors)
    ax.set_xlabel('Target Floor [GW]')
    ax.set_ylabel('Cost Savings [$B]')
    ax.set_title('c) Savings from adding solar', loc='left', fontweight='bold')

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, savings_pct)):
        if pct > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{pct:.0f}%', ha='center', fontsize=7)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'figure_battery_storage.pdf')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    print("=" * 70)
    print("BATTERY ANALYSIS - WIND vs COMBINED WIND+SOLAR")
    print("=" * 70)

    print("\nLoading data...")
    wind_data = load_wind_data(WIND_DIR)
    solar_data = load_solar_data(SOLAR_DIR)
    print(f"  Wind: {len(wind_data)} countries")
    print(f"  Solar: {len(solar_data)} countries")

    print("\nComputing timeseries...")
    wind_ts = get_aggregated_timeseries(wind_data, None)
    combined_ts = get_aggregated_timeseries(wind_data, solar_data)

    print(f"\nWIND ONLY:")
    print(f"  Hours: {len(wind_ts):,}")
    print(f"  Min: {wind_ts.min():.1f} GW")
    print(f"  Mean: {wind_ts.mean():.1f} GW")

    print(f"\nCOMBINED WIND+SOLAR:")
    print(f"  Hours: {len(combined_ts):,}")
    print(f"  Min: {combined_ts.min():.1f} GW")
    print(f"  Mean: {combined_ts.mean():.1f} GW")

    # Analyze wind-only
    print("\n" + "=" * 70)
    print("WIND-ONLY BATTERY REQUIREMENTS")
    print("=" * 70)

    wind_results = []
    for target in TARGET_FLOORS_WIND:
        r = analyze_storage(wind_ts, target)
        wind_results.append(r)
        cost = compute_cost(r)
        print(f"  {target} GW: {r['max_energy_gwh']:.0f} GWh, {r['max_power_gw']:.1f} GW, ${cost:.1f}B")

    # Analyze combined
    print("\n" + "=" * 70)
    print("COMBINED WIND+SOLAR BATTERY REQUIREMENTS")
    print("=" * 70)

    combined_results = []
    for target in TARGET_FLOORS_COMBINED:
        r = analyze_storage(combined_ts, target)
        combined_results.append(r)
        cost = compute_cost(r)
        print(f"  {target} GW: {r['max_energy_gwh']:.0f} GWh, {r['max_power_gw']:.1f} GW, ${cost:.1f}B")

    # Comparison at same floor levels
    print("\n" + "=" * 70)
    print("COMPARISON AT SAME FLOOR LEVELS")
    print("=" * 70)

    print(f"\n{'Floor':>6} | {'Wind Cost':>12} | {'Combined Cost':>14} | {'Savings':>10} | {'Savings %':>10}")
    print("-" * 65)

    for floor in [20, 25, 30, 35, 40]:
        wind_r = next((r for r in wind_results if r['target_floor_gw'] == floor), None)
        comb_r = next((r for r in combined_results if r['target_floor_gw'] == floor), None)

        if wind_r and comb_r:
            wind_cost = compute_cost(wind_r)
            comb_cost = compute_cost(comb_r)
            savings = wind_cost - comb_cost
            savings_pct = savings / wind_cost * 100 if wind_cost > 0 else 0
            print(f"{floor:>6} GW | ${wind_cost:>10.1f}B | ${comb_cost:>12.1f}B | ${savings:>8.1f}B | {savings_pct:>8.1f}%")

    print("\nCreating figure...")
    create_comparison_figure(wind_ts, combined_ts, wind_results, combined_results, OUTPUT_DIR)

    # Save results
    all_results = []
    for r in wind_results:
        all_results.append({
            'type': 'wind_only',
            'floor_gw': r['target_floor_gw'],
            'energy_gwh': r['max_energy_gwh'],
            'power_gw': r['max_power_gw'],
            'cost_billion_usd': compute_cost(r),
            'max_duration_h': r['max_event_duration'],
        })
    for r in combined_results:
        all_results.append({
            'type': 'combined',
            'floor_gw': r['target_floor_gw'],
            'energy_gwh': r['max_energy_gwh'],
            'power_gw': r['max_power_gw'],
            'cost_billion_usd': compute_cost(r),
            'max_duration_h': r['max_event_duration'],
        })

    pd.DataFrame(all_results).to_csv(
        os.path.join(RESULTS_DIR, 'battery_requirements_comparison.csv'), index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
