#!/usr/bin/env python3
"""
Master script to reproduce all results and figures.

Usage:
    python3 run_all.py              # Run everything
    python3 run_all.py --analysis   # Only analysis (Results/ CSVs)
    python3 run_all.py --figures    # Only figures (requires Results/)

Requires data in Data/Wind/ and Data/Solar/.
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

# All scripts run from the repository root; each script uses relative paths
# via os.path.join(os.path.dirname(__file__), '..', ...)

REPO_ROOT = Path(__file__).parent

# ── Analysis scripts (produce Results/ CSVs) ──────────────────────────────
# Order matters: some figure scripts also produce results
ANALYSIS_SCRIPTS = [
    ("Core analysis (multi-year)",          "Code/multi_year_analysis.py"),
    ("Monthly analysis",                    "Code/monthly_analysis.py"),
    ("Marginal diversification",            "Code/marginal_diversification.py"),
    ("Methodological improvements",         "Code/methodological_improvements.py"),
    ("Tail dependence (copula)",            "Code/tail_dependence.py"),
    ("Network analysis",                    "Code/network_analysis.py"),
]

# ── Scripts that produce both results AND figures ─────────────────────────
COMBINED_SCRIPTS = [
    ("Dunkelflaute analysis + Fig. 6",      "Code/dunkelflaute_analysis.py"),
    ("Regional clustering + Fig. 4",        "Code/regional_clustering_combined.py"),
    ("Combined wind-solar + Fig. 5",        "Code/combined_analysis_final.py"),
    ("Battery storage + Suppl. Fig. 5",     "Code/battery_baseload_combined.py"),
    ("Wavelet coherence + Suppl. Fig. 4",   "Code/wavelet_coherence.py"),
]

# ── Pure figure scripts (read from Results/) ──────────────────────────────
FIGURE_SCRIPTS = [
    ("Fig. 1 — Baseload evolution",         "Code/create_figure1_baseload.py"),
    ("Fig. 2 — Diversification benefit",    "Code/create_figure2_diversification.py"),
    ("Fig. 3 — Correlation map",            "Code/create_figure3_map.py"),
    ("Suppl. Fig. 1 — Seasonal",            "Code/create_figure3_seasonal.py"),
]


def run_script(description, script_path):
    """Run a single Python script and report success/failure."""
    full_path = REPO_ROOT / script_path
    if not full_path.exists():
        print(f"  ⚠  SKIP  {description}: {script_path} not found")
        return False

    print(f"  ▶  {description} ...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"  ✓  {description}  ({elapsed:.1f}s)")
        return True
    else:
        print(f"  ✗  {description}  FAILED ({elapsed:.1f}s)")
        # Print last 10 lines of stderr for debugging
        err_lines = result.stderr.strip().split('\n')
        for line in err_lines[-10:]:
            print(f"     {line}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Reproduce all results and figures.")
    parser.add_argument("--analysis", action="store_true", help="Run analysis scripts only")
    parser.add_argument("--figures", action="store_true", help="Run figure scripts only")
    args = parser.parse_args()

    run_analysis = not args.figures  # run unless --figures only
    run_figures = not args.analysis  # run unless --analysis only

    print("=" * 70)
    print("  Renewable Baseload — Full Reproduction Pipeline")
    print("=" * 70)

    # Check data exists
    wind_dir = REPO_ROOT / "Data" / "Wind"
    solar_dir = REPO_ROOT / "Data" / "Solar"
    if not wind_dir.exists() or not list(wind_dir.glob("*.csv")):
        print("\n  ERROR: No wind data found in Data/Wind/")
        print("  Run the download scripts first (see README.md)")
        sys.exit(1)

    total, passed, failed = 0, 0, 0

    if run_analysis:
        print("\n── Analysis scripts ─────────────────────────────────────────")
        for desc, path in ANALYSIS_SCRIPTS:
            total += 1
            if run_script(desc, path):
                passed += 1
            else:
                failed += 1

    if run_analysis and run_figures:
        print("\n── Combined analysis + figure scripts ──────────────────────")
        for desc, path in COMBINED_SCRIPTS:
            total += 1
            if run_script(desc, path):
                passed += 1
            else:
                failed += 1

    if run_figures:
        print("\n── Figure-only scripts ─────────────────────────────────────")
        for desc, path in FIGURE_SCRIPTS:
            total += 1
            if run_script(desc, path):
                passed += 1
            else:
                failed += 1

    print("\n" + "=" * 70)
    print(f"  Done: {passed}/{total} succeeded, {failed} failed")
    if failed:
        print("  Check errors above. Fig. 3 (map) requires cartopy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
