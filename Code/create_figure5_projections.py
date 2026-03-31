# -*- coding: utf-8 -*-
"""
Figure 5: Growth & Projections

Publication-quality figure for Nature Communications showing:
- Panel A: Historical mean production (2015-2024)
- Panel B: Historical baseload trend
- Panel C: Projection to 2030/2050 with EU targets

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "MultiYear")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nature Communications style settings
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

# Color scheme
COLORS = {
    'historical': '#2166ac',
    'projection': '#d6604d',
    'target': '#5aae61',
    'uncertainty': '#92c5de',
    'baseload': '#b2182b',
}

# EU Wind Energy Targets (approximate)
# Source: REPowerEU, EU Green Deal
EU_TARGETS = {
    2030: {'capacity': 450, 'description': 'REPowerEU target'},  # GW
    2050: {'capacity': 1000, 'description': 'EU climate neutrality'},  # GW
}


def load_baseload_data():
    """Load baseload statistics."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'baseload_by_year.csv'))
    return df


def create_figure5(baseload_df):
    """Create Figure 5: Growth & Projections multi-panel figure."""

    # Create figure
    fig = plt.figure(figsize=(7.08, 4.72))  # 180mm x 120mm

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.10, right=0.95, top=0.92, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    years = baseload_df['year'].values
    means = baseload_df['mean'].values / 1000  # GW
    mins = baseload_df['min'].values / 1000  # GW

    # =========================================================================
    # Panel A: Mean Production Growth
    # =========================================================================
    ax_a.bar(years, means, color=COLORS['historical'], alpha=0.8, edgecolor='white')

    # Fit trend
    z_mean = np.polyfit(years - 2015, means, 1)
    p_mean = np.poly1d(z_mean)
    ax_a.plot(years, p_mean(years - 2015), '--', color=COLORS['projection'],
              linewidth=2, label=f'+{z_mean[0]:.1f} GW/year')

    ax_a.set_xlabel('Year')
    ax_a.set_ylabel('Mean Production [GW]')
    ax_a.set_title('a) Mean Production Growth', loc='left', fontweight='bold')
    ax_a.legend(loc='upper left', frameon=False)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Growth annotation
    total_growth = means[-1] - means[0]
    pct_growth = (means[-1] / means[0] - 1) * 100
    ax_a.annotate(f'+{pct_growth:.0f}% in 10 years',
                  xy=(2024, means[-1]), xytext=(-30, 10),
                  textcoords='offset points', fontsize=7,
                  ha='right', fontweight='bold', color=COLORS['historical'])

    # =========================================================================
    # Panel B: Baseload (Minimum) Growth
    # =========================================================================
    ax_b.bar(years, mins, color=COLORS['baseload'], alpha=0.8, edgecolor='white')

    # Fit trend
    z_min = np.polyfit(years - 2015, mins, 1)
    p_min = np.poly1d(z_min)
    ax_b.plot(years, p_min(years - 2015), '--', color=COLORS['projection'],
              linewidth=2, label=f'+{z_min[0]:.2f} GW/year')

    ax_b.set_xlabel('Year')
    ax_b.set_ylabel('Minimum Production [GW]')
    ax_b.set_title('b) Baseload Growth', loc='left', fontweight='bold')
    ax_b.legend(loc='upper left', frameon=False)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Key finding
    ax_b.text(0.95, 0.95, 'Wind never stops:\nmin 5.5-11 GW',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=7,
              fontweight='bold', color=COLORS['baseload'],
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLORS['baseload'], alpha=0.9))

    # =========================================================================
    # Panel C: Projection to 2050
    # =========================================================================
    # Historical data
    ax_c.bar(years, means, color=COLORS['historical'], alpha=0.8,
             edgecolor='white', label='Historical mean production')

    # Project forward using linear trend
    future_years = np.arange(2025, 2051)
    projected_mean = p_mean(future_years - 2015)

    # Uncertainty band (widen over time)
    uncertainty = np.linspace(0, 30, len(future_years))  # Increasing uncertainty
    ax_c.fill_between(future_years,
                       projected_mean - uncertainty,
                       projected_mean + uncertainty,
                       color=COLORS['uncertainty'], alpha=0.3,
                       label='Projection uncertainty')
    ax_c.plot(future_years, projected_mean, '--', color=COLORS['projection'],
              linewidth=2, label='Projected trend')

    # Add EU targets as markers
    # Estimate mean production from capacity (assuming ~30% capacity factor)
    capacity_factor = 0.30

    for year, target in EU_TARGETS.items():
        estimated_mean = target['capacity'] * capacity_factor
        ax_c.scatter([year], [estimated_mean], s=100, marker='*',
                     color=COLORS['target'], edgecolors='white',
                     linewidths=0.5, zorder=5)
        ax_c.annotate(f"EU {year}: {target['capacity']} GW\n({estimated_mean:.0f} GW mean)",
                      xy=(year, estimated_mean),
                      xytext=(10 if year == 2030 else -10, 20),
                      textcoords='offset points',
                      fontsize=6, ha='left' if year == 2030 else 'right',
                      arrowprops=dict(arrowstyle='->', color=COLORS['target'],
                                      connectionstyle='arc3,rad=0.2'))

    ax_c.set_xlabel('Year')
    ax_c.set_ylabel('Wind Production [GW]')
    ax_c.set_title('c) Wind Energy Trajectory to 2050', loc='left', fontweight='bold')
    ax_c.set_xlim(2014, 2052)
    ax_c.set_ylim(0, 350)
    ax_c.legend(loc='upper left', frameon=False, fontsize=6)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Add vertical line separating historical/projected
    ax_c.axvline(2024.5, color='gray', linestyle=':', alpha=0.5)
    ax_c.text(2023, 320, 'Historical', ha='right', fontsize=6, color='gray')
    ax_c.text(2026, 320, 'Projected', ha='left', fontsize=6, color='gray')

    # Policy relevance note
    ax_c.text(0.98, 0.05,
              'Projections based on linear extrapolation of 2015-2024 trends.\n'
              'EU targets require accelerated deployment.',
              transform=ax_c.transAxes, ha='right', va='bottom', fontsize=5,
              style='italic', color='gray')

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'figure5_projections.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")

    return output_path


def main():
    print("=" * 60)
    print("CREATING FIGURE 5: GROWTH & PROJECTIONS")
    print("=" * 60)

    print("\nLoading baseload data...")
    baseload_df = load_baseload_data()
    print(baseload_df[['year', 'mean', 'min']].to_string(index=False))

    print("\nCreating figure...")
    output_path = create_figure5(baseload_df)

    print("\n" + "=" * 60)
    print("FIGURE 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"Finished at {datetime.now()}")
