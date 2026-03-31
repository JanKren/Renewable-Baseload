# -*- coding: utf-8 -*-
"""
Wavelet Coherence & Scale-Resolved Correlation Analysis

Replaces the original time-averaged wavelet coherence (which converged
to ~1.0 at long periods for all pairs) with two complementary analyses:

  Panel a: Scale-resolved Pearson correlation using bandpass filtering.
           Each curve shows, for one country pair, how the correlation
           depends on the timescale of wind fluctuations.  Nearby pairs
           remain correlated at synoptic scales (2-7 days); distant pairs
           lose correlation below ~1 week.

  Panel b: Timescale-specific correlation summary for three canonical
           bands (sub-synoptic: 6-48 h, synoptic: 2-7 d, seasonal: >30 d)
           versus overall Pearson r — quantifying the scale-dependent
           structure that the time-averaged coherence obscured.

Authors: Zajec & Kren
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
from datetime import datetime
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Data", "Wind")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Figures_Paper")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results", "Wavelet")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Country pairs spanning the full correlation range
KEY_PAIRS = [
    ('BE', 'NL'),   # High correlation, adjacent
    ('NO', 'SE'),   # High correlation, Nordic
    ('DE', 'FR'),   # Moderate-high correlation, neighbors
    ('DE', 'ES'),   # Low correlation, distant
    ('GB', 'GR'),   # Very low correlation, very distant
    ('PT', 'FI'),   # Near-zero correlation, opposite ends
]

PAIR_LABELS = {
    ('BE', 'NL'): 'BE-NL (adjacent)',
    ('NO', 'SE'): 'NO-SE (Nordic)',
    ('DE', 'FR'): 'DE-FR (neighbors)',
    ('DE', 'ES'): 'DE-ES (distant)',
    ('GB', 'GR'): 'GB-GR (very distant)',
    ('PT', 'FI'): 'PT-FI (opposite ends)',
}

# Color palette: warm (nearby) to cool (distant)
PAIR_COLORS = {
    ('BE', 'NL'): '#d73027',
    ('NO', 'SE'): '#fc8d59',
    ('DE', 'FR'): '#fee08b',
    ('DE', 'ES'): '#91bfdb',
    ('GB', 'GR'): '#4575b4',
    ('PT', 'FI'): '#313695',
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
    'legend.fontsize': 6.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
})


# =============================================================================
# Data loading
# =============================================================================

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


def get_aligned_pair(country_data, c1, c2, years=None):
    """Get aligned hourly data for a country pair across one or more years."""
    df1 = country_data[c1].copy()
    df2 = country_data[c2].copy()

    if years is not None:
        df1 = df1[df1.index.year.isin(years)]
        df2 = df2[df2.index.year.isin(years)]

    # Resample to hourly
    h1 = df1['wind'].resample('h').mean()
    h2 = df2['wind'].resample('h').mean()

    # Common index, drop NaN
    common_idx = h1.index.intersection(h2.index)
    x = h1.loc[common_idx].values
    y = h2.loc[common_idx].values
    valid = ~(np.isnan(x) | np.isnan(y))
    return x[valid], y[valid]


# =============================================================================
# Scale-resolved correlation (bandpass approach)
# =============================================================================

def bandpass_filter(data, fs, low_period, high_period, order=4):
    """
    Butterworth bandpass filter.

    Parameters
    ----------
    data : array
        Time series.
    fs : float
        Sampling frequency (1 for hourly data).
    low_period, high_period : float
        Band limits in hours (high_period > low_period).
        The filter passes fluctuations with period in [low_period, high_period].
    order : int
        Filter order.
    """
    nyq = 0.5 * fs
    low_freq = 1.0 / high_period   # lower frequency (longer period)
    high_freq = 1.0 / low_period   # higher frequency (shorter period)

    # Clamp to valid range
    low_freq = max(low_freq, 1e-8)
    high_freq = min(high_freq, nyq * 0.99)

    if low_freq >= high_freq:
        return np.zeros_like(data)

    b, a = signal.butter(order, [low_freq / nyq, high_freq / nyq], btype='band')
    return signal.filtfilt(b, a, data)


def lowpass_filter(data, fs, cutoff_period, order=4):
    """Butterworth low-pass filter (passes periods > cutoff_period)."""
    nyq = 0.5 * fs
    cutoff_freq = 1.0 / cutoff_period
    cutoff_freq = max(cutoff_freq, 1e-8)
    if cutoff_freq >= nyq:
        return data
    b, a = signal.butter(order, cutoff_freq / nyq, btype='low')
    return signal.filtfilt(b, a, data)


def scale_resolved_correlation(x, y, fs=1.0, n_bands=30):
    """
    Compute Pearson correlation of bandpass-filtered signals at
    logarithmically spaced center periods.

    For each center period T_c, we filter both signals to the band
    [T_c / sqrt(2), T_c * sqrt(2)] (one-octave bandwidth) and compute
    the Pearson correlation of the filtered signals.

    Parameters
    ----------
    x, y : array
        Aligned time series (hourly).
    fs : float
        Sampling frequency.
    n_bands : int
        Number of center periods.

    Returns
    -------
    center_periods : array
        Center periods in hours.
    correlations : array
        Pearson r at each scale.
    """
    min_period = 8        # hours
    max_period = 60 * 24  # 60 days (in hours)

    center_periods = np.logspace(np.log10(min_period), np.log10(max_period), n_bands)
    correlations = np.full(n_bands, np.nan)

    factor = np.sqrt(2)  # one-octave bandwidth

    for i, Tc in enumerate(center_periods):
        low_p = Tc / factor
        high_p = Tc * factor

        # Ensure high_period doesn't exceed data length
        if high_p > len(x) * 0.4:
            # Use lowpass for the longest scales
            xf = lowpass_filter(x, fs, low_p, order=3)
            yf = lowpass_filter(y, fs, low_p, order=3)
        else:
            xf = bandpass_filter(x, fs, low_p, high_p, order=3)
            yf = bandpass_filter(y, fs, low_p, high_p, order=3)

        # Remove edge transients (10% on each side)
        edge = int(0.1 * len(xf))
        xf = xf[edge:-edge]
        yf = yf[edge:-edge]

        if np.std(xf) > 1e-10 and np.std(yf) > 1e-10:
            correlations[i] = np.corrcoef(xf, yf)[0, 1]

    return center_periods, correlations


# =============================================================================
# Figure creation
# =============================================================================

def create_figure(country_data, key_pairs, output_dir):
    """Create the improved wavelet/scale-resolved coherence figure."""

    # Use full decade for robust statistics
    years = list(range(2015, 2025))

    # ---- Compute all scale-resolved correlations ----
    results = {}
    for c1, c2 in key_pairs:
        print(f"  Computing scale-resolved correlation for {c1}-{c2}...")
        x, y = get_aligned_pair(country_data, c1, c2, years=years)
        pearson_r = np.corrcoef(x, y)[0, 1]
        periods, corrs = scale_resolved_correlation(x, y, n_bands=35)
        results[(c1, c2)] = {
            'pearson_r': pearson_r,
            'periods': periods,
            'correlations': corrs
        }
        print(f"    Pearson r = {pearson_r:.3f}, "
              f"min scale-r = {np.nanmin(corrs):.3f}, "
              f"max scale-r = {np.nanmax(corrs):.3f}")

    # ---- Create figure ----
    fig = plt.figure(figsize=(7.08, 3.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1], wspace=0.4)

    # =====================
    # Panel a: Scale-resolved correlation curves
    # =====================
    ax_a = fig.add_subplot(gs[0])

    for c1, c2 in key_pairs:
        r = results[(c1, c2)]
        color = PAIR_COLORS[(c1, c2)]
        label = f"{c1}-{c2} (r = {r['pearson_r']:.2f})"
        ax_a.semilogx(r['periods'], r['correlations'],
                       '-', linewidth=1.4, color=color, label=label)

    # Reference timescales
    for p, lbl in [(24, '1 day'), (168, '1 week'), (720, '4.3 weeks')]:
        ax_a.axvline(p, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax_a.text(p, 1.03, lbl, fontsize=5.5, ha='center', color='0.4',
                  transform=ax_a.get_xaxis_transform())

    # Shade synoptic band
    ax_a.axvspan(48, 168, alpha=0.06, color='steelblue', zorder=0)
    ax_a.text(np.sqrt(48 * 168), 0.05, 'synoptic\n(2-7 d)',
              fontsize=5.5, ha='center', va='bottom', color='steelblue',
              transform=ax_a.get_xaxis_transform())

    ax_a.set_xlabel('Period [hours]')
    ax_a.set_ylabel('Pearson correlation')
    ax_a.set_xlim(8, 1500)
    ax_a.set_ylim(-0.15, 1.05)
    ax_a.axhline(0, color='black', linewidth=0.3)
    ax_a.legend(loc='upper left', framealpha=0.9, borderpad=0.4,
                handlelength=1.5)
    ax_a.set_title('a', fontsize=10, fontweight='bold', loc='left', x=-0.08)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # =====================
    # Panel b: Band-averaged correlation summary
    # =====================
    ax_b = fig.add_subplot(gs[1])

    # Define three bands
    bands = {
        'Sub-synoptic\n(6-48 h)': (6, 48),
        'Synoptic\n(2-7 d)': (48, 168),
        'Planetary\n(>30 d)': (720, 1500),
    }

    band_corrs = {bname: [] for bname in bands}
    pair_labels_short = []

    for c1, c2 in key_pairs:
        r = results[(c1, c2)]
        pair_labels_short.append(f"{c1}-{c2}")
        for bname, (lo, hi) in bands.items():
            mask = (r['periods'] >= lo) & (r['periods'] <= hi)
            if mask.any():
                band_corrs[bname].append(np.nanmean(r['correlations'][mask]))
            else:
                band_corrs[bname].append(np.nan)

    x_pos = np.arange(len(key_pairs))
    width = 0.25
    band_colors = ['#fee08b', '#91bfdb', '#4575b4']

    for i, (bname, vals) in enumerate(band_corrs.items()):
        offset = (i - 1) * width
        bars = ax_b.bar(x_pos + offset, vals, width, label=bname,
                        color=band_colors[i], edgecolor='white', linewidth=0.3)

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(pair_labels_short, rotation=50, ha='right', fontsize=6)
    ax_b.set_ylabel('Mean correlation in band')
    ax_b.set_ylim(-0.15, 1.05)
    ax_b.axhline(0, color='black', linewidth=0.3)
    ax_b.legend(loc='upper right', framealpha=0.9, borderpad=0.3,
                fontsize=5.5, handlelength=1.0)
    ax_b.set_title('b', fontsize=10, fontweight='bold', loc='left', x=-0.12)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    plt.tight_layout()

    # ---- Save ----
    out_png = os.path.join(output_dir, 'figure_wavelet_coherence.png')
    out_pdf = os.path.join(output_dir, 'figure_wavelet_coherence.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    # ---- Save results CSV ----
    rows = []
    for c1, c2 in key_pairs:
        r = results[(c1, c2)]
        for bname, (lo, hi) in bands.items():
            mask = (r['periods'] >= lo) & (r['periods'] <= hi)
            mean_r = np.nanmean(r['correlations'][mask]) if mask.any() else np.nan
            rows.append({
                'pair': f"{c1}-{c2}",
                'pearson_r': r['pearson_r'],
                'band': bname.replace('\n', ' '),
                'band_lo_h': lo,
                'band_hi_h': hi,
                'mean_band_correlation': mean_r
            })
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'scale_resolved_correlations.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("SCALE-RESOLVED CORRELATION ANALYSIS (Supplementary Figure 4)")
    print("=" * 70)

    print("\nLoading country data...")
    country_data = load_all_country_data(DATA_DIR)
    print(f"Loaded {len(country_data)} countries")

    valid_pairs = []
    for c1, c2 in KEY_PAIRS:
        if c1 in country_data and c2 in country_data:
            valid_pairs.append((c1, c2))
        else:
            print(f"  Skipping {c1}-{c2}: missing data")

    print(f"\nAnalyzing {len(valid_pairs)} country pairs "
          f"across the full decade (2015-2024)...")

    results = create_figure(country_data, valid_pairs, OUTPUT_DIR)

    # ---- Print summary ----
    print("\n" + "-" * 60)
    print("SCALE-RESOLVED CORRELATION SUMMARY (decade: 2015-2024)")
    print("-" * 60)

    bands_summary = {
        'Sub-synoptic (6-48h)': (6, 48),
        'Synoptic (2-7d)': (48, 168),
        'Planetary (>30d)': (720, 1500)
    }

    header = f"{'Pair':<10} {'Pearson r':>9}"
    for bname in bands_summary:
        header += f"  {bname:>22}"
    print(header)
    print("-" * len(header))

    for c1, c2 in valid_pairs:
        r = results[(c1, c2)]
        line = f"{c1}-{c2:<6} {r['pearson_r']:>9.3f}"
        for bname, (lo, hi) in bands_summary.items():
            mask = (r['periods'] >= lo) & (r['periods'] <= hi)
            val = np.nanmean(r['correlations'][mask]) if mask.any() else np.nan
            line += f"  {val:>22.3f}"
        print(line)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print(f"Started at {datetime.now()}")
    main()
    print(f"\nFinished at {datetime.now()}")
