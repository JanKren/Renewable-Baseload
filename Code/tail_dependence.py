# -*- coding: utf-8 -*-
"""
Tail Dependence Analysis via Copulas

Analyzes whether European wind production is MORE correlated during
extreme low-wind events than during normal conditions.

Methods:
- Fit Gaussian copula (baseline, λL = 0)
- Fit Clayton copula (lower tail dependence)
- Fit Gumbel copula (upper tail dependence)
- Compare tail dependence coefficients

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "TailDependence")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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
    'highlight': '#d6604d',
    'neutral': '#878787',
}

# Geographic ordering
GEOGRAPHIC_ORDER = [
    'PT', 'ES', 'IE', 'GB', 'FR', 'BE', 'NL', 'LU', 'DE', 'DK',
    'NO', 'SE', 'FI', 'EE', 'LV', 'LT', 'PL', 'CZ', 'SK', 'AT',
    'CH', 'SI', 'HR', 'HU', 'RO', 'BG', 'RS', 'GR', 'IT'
]


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


def align_country_pair(data1, data2):
    """Align two country time series to common hourly index."""
    # Resample to hourly
    hourly1 = data1['wind'].resample('h').mean()
    hourly2 = data2['wind'].resample('h').mean()

    # Find common index
    common_idx = hourly1.index.intersection(hourly2.index)

    if len(common_idx) < 1000:
        return None, None

    x = hourly1.loc[common_idx].values
    y = hourly2.loc[common_idx].values

    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]

    if len(x) < 1000:
        return None, None

    return x, y


def empirical_cdf_transform(x):
    """Transform data to uniform [0, 1] using empirical CDF."""
    n = len(x)
    ranks = stats.rankdata(x)
    # Use (rank - 0.5) / n to avoid 0 and 1
    u = (ranks - 0.5) / n
    return u


def fit_clayton_copula(u, v):
    """
    Fit Clayton copula and estimate lower tail dependence.

    Clayton copula: C(u,v) = (u^(-θ) + v^(-θ) - 1)^(-1/θ)
    Lower tail dependence: λL = 2^(-1/θ)
    """

    def neg_log_likelihood(theta):
        if theta <= 0:
            return 1e10

        # Clayton copula density
        try:
            term1 = np.log(1 + theta)
            term2 = -(1 + theta) * (np.log(u) + np.log(v))
            term3 = -(2 + 1/theta) * np.log(u**(-theta) + v**(-theta) - 1)
            ll = np.sum(term1 + term2 + term3)
            return -ll
        except:
            return 1e10

    # Optimize
    result = minimize_scalar(neg_log_likelihood, bounds=(0.01, 20), method='bounded')
    theta = result.x

    # Lower tail dependence
    lambda_L = 2 ** (-1 / theta) if theta > 0 else 0

    return theta, lambda_L


def fit_gumbel_copula(u, v):
    """
    Fit Gumbel copula and estimate upper tail dependence.

    Gumbel copula: C(u,v) = exp(-[(-log u)^θ + (-log v)^θ]^(1/θ))
    Upper tail dependence: λU = 2 - 2^(1/θ)
    """

    def neg_log_likelihood(theta):
        if theta < 1:
            return 1e10

        try:
            # Avoid log(0)
            u_safe = np.clip(u, 1e-10, 1 - 1e-10)
            v_safe = np.clip(v, 1e-10, 1 - 1e-10)

            log_u = -np.log(u_safe)
            log_v = -np.log(v_safe)

            A = (log_u ** theta + log_v ** theta) ** (1 / theta)

            term1 = -A
            term2 = (theta - 1) * (np.log(log_u) + np.log(log_v))
            term3 = (1/theta - 2) * np.log(log_u ** theta + log_v ** theta)
            term4 = np.log(A + theta - 1)

            ll = np.sum(term1 + term2 + term3 + term4)
            return -ll
        except:
            return 1e10

    # Optimize
    result = minimize_scalar(neg_log_likelihood, bounds=(1.01, 20), method='bounded')
    theta = result.x

    # Upper tail dependence
    lambda_U = 2 - 2 ** (1 / theta) if theta > 1 else 0

    return theta, lambda_U


def compute_empirical_tail_dependence(u, v, q=0.1):
    """
    Compute empirical lower tail dependence.

    λL ≈ P(V < q | U < q) as q → 0
    """
    below_u = u < q
    below_v = v < q

    if below_u.sum() == 0:
        return 0

    # Conditional probability
    lambda_L_empirical = (below_u & below_v).sum() / below_u.sum()

    return lambda_L_empirical


def analyze_all_pairs(country_data):
    """Analyze tail dependence for all country pairs."""

    countries = [c for c in GEOGRAPHIC_ORDER if c in country_data]
    n_countries = len(countries)

    print(f"\nAnalyzing {n_countries} countries, {n_countries * (n_countries - 1) // 2} pairs")

    results = []

    for i, c1 in enumerate(countries):
        for j, c2 in enumerate(countries):
            if i >= j:
                continue

            print(f"  {c1}-{c2}...", end=' ', flush=True)

            x, y = align_country_pair(country_data[c1], country_data[c2])

            if x is None:
                print("insufficient data")
                continue

            # Pearson correlation
            pearson_r = np.corrcoef(x, y)[0, 1]

            # Transform to uniform marginals
            u = empirical_cdf_transform(x)
            v = empirical_cdf_transform(y)

            # Fit copulas
            try:
                clayton_theta, lambda_L = fit_clayton_copula(u, v)
            except:
                clayton_theta, lambda_L = np.nan, np.nan

            try:
                gumbel_theta, lambda_U = fit_gumbel_copula(u, v)
            except:
                gumbel_theta, lambda_U = np.nan, np.nan

            # Empirical tail dependence (at 10th percentile)
            lambda_L_empirical = compute_empirical_tail_dependence(u, v, q=0.10)

            results.append({
                'country1': c1,
                'country2': c2,
                'n_obs': len(x),
                'pearson_r': pearson_r,
                'clayton_theta': clayton_theta,
                'lambda_L': lambda_L,
                'gumbel_theta': gumbel_theta,
                'lambda_U': lambda_U,
                'lambda_L_empirical': lambda_L_empirical
            })

            print(f"r={pearson_r:.2f}, λL={lambda_L:.3f}")

    return pd.DataFrame(results), countries


def create_tail_dependence_figure(results_df, countries, output_dir):
    """Create Figure 5: Tail Dependence Analysis."""

    fig = plt.figure(figsize=(10, 9))

    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1],
                          hspace=0.45, wspace=0.35,
                          left=0.08, right=0.95, top=0.95, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, :])   # Heatmap — full width, top row
    ax_b = fig.add_subplot(gs[1, 0])   # Scatter: λL vs Pearson
    ax_d = fig.add_subplot(gs[1, 1])   # Key pairs comparison

    # =========================================================================
    # Panel A: Lower Tail Dependence Heatmap
    # =========================================================================
    n = len(countries)
    lambda_matrix = np.zeros((n, n))
    lambda_matrix[:] = np.nan

    for _, row in results_df.iterrows():
        i = countries.index(row['country1'])
        j = countries.index(row['country2'])
        lambda_matrix[i, j] = row['lambda_L']
        lambda_matrix[j, i] = row['lambda_L']

    # Diagonal
    np.fill_diagonal(lambda_matrix, 1.0)

    mask = np.triu(np.ones_like(lambda_matrix, dtype=bool), k=1)

    sns.heatmap(lambda_matrix, ax=ax_a, mask=mask,
                cmap='YlOrRd', vmin=0, vmax=0.8,
                square=False, linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Lower Tail Dependence λL', 'shrink': 0.6},
                xticklabels=countries, yticklabels=countries)

    ax_a.set_title('a) Lower Tail Dependence', loc='left', fontweight='bold')
    # Fix tick alignment: seaborn places ticks correctly, just style them
    plt.setp(ax_a.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor', fontsize=10)
    plt.setp(ax_a.get_yticklabels(), rotation=0, fontsize=10)

    # =========================================================================
    # Panel B: λL vs Pearson Correlation
    # =========================================================================
    valid = results_df['lambda_L'].notna()

    ax_b.scatter(results_df.loc[valid, 'pearson_r'],
                results_df.loc[valid, 'lambda_L'],
                alpha=0.5, s=20, c=COLORS['primary'], edgecolors='white', linewidths=0.3)

    # Add Gaussian copula reference (λL = 0 for all r)
    ax_b.axhline(0, color=COLORS['secondary'], linestyle='--', linewidth=1.5,
                 label='Gaussian copula (λL=0)')

    # Trend line
    z = np.polyfit(results_df.loc[valid, 'pearson_r'], results_df.loc[valid, 'lambda_L'], 1)
    x_line = np.linspace(results_df['pearson_r'].min(), results_df['pearson_r'].max(), 100)
    ax_b.plot(x_line, np.poly1d(z)(x_line), '-', color=COLORS['highlight'], linewidth=1.5,
              label=f'Trend: λL = {z[0]:.2f}r + {z[1]:.2f}')

    ax_b.set_xlabel('Pearson Correlation r')
    ax_b.set_ylabel('Lower Tail Dependence λL')
    ax_b.set_title('b) Tail Dependence vs Linear Correlation', loc='left', fontweight='bold')
    ax_b.legend(loc='upper left', frameon=False, fontsize=6)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # Add key finding
    mean_lambda = results_df['lambda_L'].mean()
    ax_b.text(0.95, 0.95, f'Mean λL = {mean_lambda:.3f}\n(Gaussian: λL = 0)',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=7,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLORS['secondary'], alpha=0.9))

    # =========================================================================
    # Panel C: Key Pairs Comparison (was Panel D)
    # =========================================================================
    # Select interesting pairs: high/low correlation, high/low tail dependence
    key_pairs = results_df.nlargest(6, 'lambda_L')[['country1', 'country2', 'pearson_r', 'lambda_L']]

    y_pos = np.arange(len(key_pairs))
    pairs_labels = [f"{row['country1']}-{row['country2']}" for _, row in key_pairs.iterrows()]

    ax_d.barh(y_pos - 0.15, key_pairs['pearson_r'], 0.3, label='Pearson r', color=COLORS['primary'], alpha=0.8)
    ax_d.barh(y_pos + 0.15, key_pairs['lambda_L'], 0.3, label='λL', color=COLORS['secondary'], alpha=0.8)

    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(pairs_labels)
    ax_d.set_xlabel('Coefficient Value')
    ax_d.set_title('c) Highest Tail Dependence Pairs', loc='left', fontweight='bold')
    ax_d.legend(loc='upper center', frameon=True, fancybox=False, edgecolor='lightgray')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    # Save figure
    output_path = os.path.join(output_dir, 'figure5_tail_dependence.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    print(f"Saved: {output_path.replace('.png', '.pdf')}")


def main():
    print("=" * 70)
    print("TAIL DEPENDENCE ANALYSIS VIA COPULAS")
    print("=" * 70)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    print("\nAnalyzing all country pairs...")
    results_df, countries = analyze_all_pairs(country_data)

    print("\n" + "-" * 50)
    print("SUMMARY STATISTICS")
    print("-" * 50)
    print(f"Total pairs analyzed: {len(results_df)}")
    print(f"Mean Pearson r: {results_df['pearson_r'].mean():.3f}")
    print(f"Mean lower tail dependence λL: {results_df['lambda_L'].mean():.3f}")
    print(f"Pairs with λL > 0.3: {(results_df['lambda_L'] > 0.3).sum()}")
    print(f"Pairs with λL > 0.5: {(results_df['lambda_L'] > 0.5).sum()}")

    print("\nHighest tail dependence pairs:")
    print(results_df.nlargest(10, 'lambda_L')[['country1', 'country2', 'pearson_r', 'lambda_L']].to_string(index=False))

    print("\nCreating figure...")
    create_tail_dependence_figure(results_df, countries, OUTPUT_DIR)

    print("\nSaving results...")
    results_df.to_csv(os.path.join(RESULTS_DIR, 'tail_dependence_all_pairs.csv'), index=False)
    print(f"Saved: {RESULTS_DIR}/tail_dependence_all_pairs.csv")

    print("\n" + "=" * 70)
    print("TAIL DEPENDENCE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
