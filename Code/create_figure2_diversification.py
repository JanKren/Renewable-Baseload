# -*- coding: utf-8 -*-
"""
Figure 2: Diversification Benefit

Publication-quality figure for Nature Communications showing:
- Panel A: Individual country CV vs aggregated CV (bar comparison)
- Panel B: 10-year stability of diversification benefit
- Panel C: Portfolio effect conceptual diagram

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import glob
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
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

# Color scheme (colorblind-friendly)
COLORS = {
    'individual': '#d6604d',   # Red/coral
    'aggregated': '#2166ac',   # Blue
    'reduction': '#5aae61',    # Green
    'neutral': '#878787',      # Gray
    'fill': '#92c5de',         # Light blue
    'light_red': '#f4a582',
    'light_green': '#a6dba0',
}


def load_diversification_data():
    """Load diversification statistics."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'diversification_by_year.csv'))
    return df


def create_figure2(div_df):
    """Create Figure 2: Diversification Benefit multi-panel figure."""

    # Create figure: 180mm wide, ~100mm tall
    fig = plt.figure(figsize=(7.08, 3.94))  # 180mm x 100mm

    # Create grid
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1],
                          wspace=0.5,
                          left=0.08, right=0.96, top=0.88, bottom=0.15)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # =========================================================================
    # Panel A: Individual vs Aggregated CV (latest year + average)
    # =========================================================================
    # Use 2024 data and 10-year average
    latest = div_df[div_df['year'] == 2024].iloc[0]
    avg_individual = div_df['mean_individual_cv'].mean()
    avg_aggregated = div_df['aggregated_cv'].mean()

    categories = ['2024', '10-Year\nAverage']
    individual_vals = [latest['mean_individual_cv'], avg_individual]
    aggregated_vals = [latest['aggregated_cv'], avg_aggregated]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax_a.bar(x - width/2, individual_vals, width,
                     label='Individual countries', color=COLORS['individual'], alpha=0.8)
    bars2 = ax_a.bar(x + width/2, aggregated_vals, width,
                     label='European aggregate', color=COLORS['aggregated'], alpha=0.8)

    # Add reduction arrows
    for i, (ind, agg) in enumerate(zip(individual_vals, aggregated_vals)):
        reduction = (1 - agg/ind) * 100
        mid_y = (ind + agg) / 2

        # Draw arrow
        ax_a.annotate('', xy=(i + width/2, agg + 0.02),
                      xytext=(i - width/2, ind - 0.02),
                      arrowprops=dict(arrowstyle='->', color=COLORS['reduction'],
                                      lw=1.5, shrinkA=0, shrinkB=0))

        # Add reduction percentage
        ax_a.text(i, mid_y, f'-{reduction:.0f}%',
                  ha='center', va='center', fontsize=8, fontweight='bold',
                  color=COLORS['reduction'],
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=COLORS['reduction'], alpha=0.9))

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(categories)
    ax_a.set_ylabel('Coefficient of Variation (CV)')
    ax_a.set_title('a) Variability Reduction', loc='left', fontweight='bold')
    ax_a.legend(loc='upper right', frameon=False, fontsize=6)
    ax_a.set_ylim(0, 1.1)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # =========================================================================
    # Panel B: 10-Year Stability of Diversification Benefit
    # =========================================================================
    years = div_df['year'].values
    reduction = div_df['cv_reduction_pct'].values

    ax_b.bar(years, reduction, color=COLORS['reduction'], alpha=0.8, edgecolor='white')

    # Add mean line
    mean_reduction = np.mean(reduction)
    ax_b.axhline(mean_reduction, color=COLORS['neutral'], linestyle='--', linewidth=1.5)

    # Add confidence band (std)
    std_reduction = np.std(reduction)
    ax_b.fill_between([years[0]-0.5, years[-1]+0.5],
                       mean_reduction - std_reduction,
                       mean_reduction + std_reduction,
                       alpha=0.2, color=COLORS['reduction'])

    ax_b.set_xlabel('Year')
    ax_b.set_ylabel('DB [%]')
    ax_b.set_title('b) DB stability', loc='left', fontweight='bold')
    ax_b.set_ylim(35, 65)
    ax_b.set_xlim(2014.5, 2024.5)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Add annotation inside plot area to avoid overlap
    ax_b.text(0.5, 0.08, f'{mean_reduction:.0f}% ± {std_reduction:.0f}%',
              transform=ax_b.transAxes, ha='center', va='bottom',
              fontsize=9, fontweight='bold', color=COLORS['reduction'],
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # =========================================================================
    # Panel C: CV vs Number of Countries with yearly error bands
    # =========================================================================
    # Import greedy algorithm from marginal_diversification
    code_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, code_dir)
    from marginal_diversification import (load_all_country_data, compute_aligned_data,
                                           greedy_portfolio_construction, compute_cv)

    # Load raw data and run greedy per year
    print("  Computing per-year greedy curves for error bands...")
    country_data = load_all_country_data(DATA_DIR)

    yearly_cv_curves = []
    for year in range(2015, 2025):
        aligned_year = compute_aligned_data(country_data, year=year)
        if len(aligned_year.columns) < 10:
            continue
        evo_year, _ = greedy_portfolio_construction(aligned_year)
        cv_curve = evo_year['portfolio_cv'].values
        yearly_cv_curves.append(cv_curve)

    # Pad to same length (max countries)
    max_len = max(len(c) for c in yearly_cv_curves)
    padded_curves = []
    for curve in yearly_cv_curves:
        if len(curve) < max_len:
            padded = np.full(max_len, np.nan)
            padded[:len(curve)] = curve
            padded_curves.append(padded)
        else:
            padded_curves.append(curve)

    cv_matrix = np.array(padded_curves)
    mean_cv = np.nanmean(cv_matrix, axis=0)
    std_cv = np.nanstd(cv_matrix, axis=0)
    n_countries_range = np.arange(1, max_len + 1)

    # Plot mean line with shaded error band
    ax_c.fill_between(n_countries_range, mean_cv - std_cv, mean_cv + std_cv,
                       alpha=0.25, color=COLORS['aggregated'], label='$\\pm$1 s.d. (10 years)')
    ax_c.plot(n_countries_range, mean_cv, 'o-', color=COLORS['aggregated'],
              linewidth=1.5, markersize=3, alpha=0.8, label='Mean CV')

    # Highlight key points
    ax_c.scatter([1], [mean_cv[0]], s=60, color=COLORS['individual'],
                 zorder=5, edgecolor='white', linewidth=1)
    ax_c.scatter([max_len], [mean_cv[-1]], s=60,
                 color=COLORS['aggregated'], zorder=5, edgecolor='white', linewidth=1)

    # Add annotations
    ax_c.annotate(f'Single country\nCV = {mean_cv[0]:.2f}',
                  xy=(1, mean_cv[0]), xytext=(12, mean_cv[0] + 0.05),
                  fontsize=6, ha='center', color=COLORS['individual'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['individual'], lw=0.5))

    last_valid = ~np.isnan(mean_cv)
    last_idx = np.where(last_valid)[0][-1]
    ax_c.annotate(f'All countries\nCV = {mean_cv[last_idx]:.2f}',
                  xy=(last_idx + 1, mean_cv[last_idx]),
                  xytext=(last_idx - 5, mean_cv[last_idx] + 0.08),
                  fontsize=6, ha='right', color=COLORS['aggregated'],
                  arrowprops=dict(arrowstyle='->', color=COLORS['aggregated'], lw=0.5))

    # Add theoretical 1/sqrt(n) reference line
    n_theory = np.linspace(1, max_len, 100)
    cv_theory = mean_cv[0] / np.sqrt(n_theory)
    ax_c.plot(n_theory, cv_theory, '--', color=COLORS['neutral'],
              linewidth=1, alpha=0.5, label='$1/\\sqrt{n}$ (uncorrelated)')

    ax_c.set_xlabel('Number of Countries')
    ax_c.set_ylabel('Portfolio CV')
    ax_c.set_title('c) Portfolio Effect', loc='left', fontweight='bold')
    ax_c.set_xlim(0, max_len + 2)
    ax_c.set_ylim(0.2, 1.0)
    ax_c.legend(loc='upper right', frameon=False, fontsize=7)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    # Add reduction annotation
    reduction_pct = (1 - mean_cv[last_idx]/mean_cv[0]) * 100
    ax_c.text(0.5, 0.15, f'{reduction_pct:.0f}% reduction',
              transform=ax_c.transAxes, ha='center', fontsize=8,
              fontweight='bold', color=COLORS['reduction'])

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, 'figure2_diversification.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")

    return output_path


def main():
    print("=" * 60)
    print("CREATING FIGURE 2: DIVERSIFICATION BENEFIT")
    print("=" * 60)

    print("\nLoading diversification data...")
    div_df = load_diversification_data()
    print(div_df.to_string(index=False))

    print("\nCreating figure...")
    output_path = create_figure2(div_df)

    print("\n" + "=" * 60)
    print("FIGURE 2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"Finished at {datetime.now()}")
