# -*- coding: utf-8 -*-
"""
Figure 3: Seasonal Patterns

Publication-quality figure for Nature Communications showing:
- Panel A: Year×Month heatmap of minimum production
- Panel B: Seasonal averages (bar chart by season)
- Panel C: Best vs Worst month time series comparison

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Monthly")
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

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

SEASON_COLORS = {
    'Winter': '#2166ac',   # Blue
    'Spring': '#5aae61',   # Green
    'Summer': '#fdae61',   # Orange
    'Autumn': '#d6604d',   # Red
}


def load_monthly_data():
    """Load monthly statistics."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'monthly_statistics.csv'))
    return df


def create_figure3(monthly_df):
    """Create Figure 3: Seasonal Patterns multi-panel figure."""

    # Create figure: 180mm wide, ~120mm tall
    fig = plt.figure(figsize=(7.08, 4.72))

    # Create grid
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.5, 1],
                          hspace=0.35, wspace=0.30,
                          left=0.08, right=0.96, top=0.92, bottom=0.10)

    ax_a = fig.add_subplot(gs[0, :])  # Heatmap spans top
    ax_b = fig.add_subplot(gs[1, 0])   # Seasonal bar chart
    ax_c = fig.add_subplot(gs[1, 1])   # CV reduction by season

    # =========================================================================
    # Panel A: Year×Month Heatmap of Minimum Production
    # =========================================================================
    # Pivot for heatmap
    pivot = monthly_df.pivot(index='month', columns='year', values='min')
    pivot = pivot / 1000  # Convert to GW

    # Create heatmap
    sns.heatmap(pivot, ax=ax_a, cmap='YlOrRd_r', annot=True, fmt='.1f',
                annot_kws={'fontsize': 6},
                cbar_kws={'label': 'Minimum Production [GW]', 'shrink': 0.8},
                linewidths=0.5, linecolor='white',
                yticklabels=MONTH_NAMES)

    ax_a.set_ylabel('Month')
    ax_a.set_xlabel('Year')
    ax_a.set_title('a) Monthly Minimum Production (2015-2024)', loc='left', fontweight='bold')

    # Rotate x labels
    ax_a.set_xticklabels(ax_a.get_xticklabels(), rotation=0)

    # =========================================================================
    # Panel B: Seasonal Averages
    # =========================================================================
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Autumn': [9, 10, 11]
    }

    season_stats = []
    for season, months in seasons.items():
        season_data = monthly_df[monthly_df['month'].isin(months)]
        season_stats.append({
            'season': season,
            'mean_production': season_data['mean'].mean() / 1000,
            'min_production': season_data['min'].mean() / 1000,
            'cv_reduction': season_data['cv_reduction'].mean()
        })

    season_df = pd.DataFrame(season_stats)

    x = np.arange(len(season_df))
    width = 0.35

    bars1 = ax_b.bar(x - width/2, season_df['mean_production'], width,
                     label='Mean', color=[SEASON_COLORS[s] for s in season_df['season']],
                     alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax_b.bar(x + width/2, season_df['min_production'], width,
                     label='Minimum', color=[SEASON_COLORS[s] for s in season_df['season']],
                     alpha=0.5, edgecolor='white', linewidth=0.5,
                     hatch='///')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(season_df['season'])
    ax_b.set_ylabel('Production [GW]')
    ax_b.set_title('b) Seasonal Production', loc='left', fontweight='bold')
    ax_b.legend(loc='upper right', frameon=False, fontsize=6)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax_b.annotate(f'{height:.0f}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 2), textcoords="offset points",
                      ha='center', va='bottom', fontsize=5)

    # =========================================================================
    # Panel C: CV Reduction by Season
    # =========================================================================
    colors = [SEASON_COLORS[s] for s in season_df['season']]
    bars = ax_c.bar(season_df['season'], season_df['cv_reduction'],
                    color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    # Add value labels inside bars (white text, placed at mid-height)
    for bar in bars:
        height = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width() / 2, height / 2,
                  f'{height:.1f}%', ha='center', va='center',
                  fontsize=6, fontweight='bold', color='white')

    # Add mean line with label above, right-aligned inside plot
    mean_cv = season_df['cv_reduction'].mean()
    ax_c.axhline(mean_cv, color='gray', linestyle='--', linewidth=1)
    ax_c.text(0.02, mean_cv + 1.5, f'Mean: {mean_cv:.1f}%', fontsize=5.5, ha='left',
              color='0.4', transform=ax_c.get_yaxis_transform())

    ax_c.set_ylabel('CV Reduction [%]')
    ax_c.set_title('c) Diversification Benefit', loc='left', fontweight='bold')
    ax_c.set_ylim(0, 65)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'figure3_seasonal.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")

    return output_path


def main():
    print("=" * 60)
    print("CREATING FIGURE 3: SEASONAL PATTERNS")
    print("=" * 60)

    print("\nLoading monthly data...")
    monthly_df = load_monthly_data()
    print(f"Loaded {len(monthly_df)} monthly records")

    print("\nCreating figure...")
    output_path = create_figure3(monthly_df)

    print("\n" + "=" * 60)
    print("FIGURE 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"Finished at {datetime.now()}")
