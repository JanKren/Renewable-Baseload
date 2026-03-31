# The Wind Is Always Blowing Somewhere in Europe

**A decade of evidence for continental-scale renewable baseload**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> Jan Kren (Paul Scherrer Institute), Boštjan Zajec (CEA Paris-Saclay), Iztok Tiselj (University of Ljubljana)
>
> Submitted to *Nature Communications* (2026)

---

## Overview

This repository contains the analysis code, processed results, and figure-generation scripts for a decade-long study (2015–2024) of European wind and solar power production. Using ~87,600 hours of actual ENTSO-E generation data from 29 countries, we show that geographic diversification reliably halves wind variability, identify where new capacity would yield the greatest reliability gains, and quantify the benefits of combined wind–solar portfolios.

### Key findings

| Finding | Value |
|---------|-------|
| Absolute minimum wind production | 5.5 GW (never zero in 87,600 hours) |
| 1st-percentile baseload | 12.5 GW [95% CI: 11.9–13.3] |
| Diversification benefit (CV reduction) | 47.8 ± 4.3% (stable over decade) |
| Marginal baseload conversion rate | 2.1% (48 GW new capacity per 1 GW baseload) |
| Interconnection dividend (wind-only) | 2.5× vs isolated regional grids |
| Interconnection dividend (wind+solar) | 3.1× vs isolated regional grids |
| Baseload improvement from adding solar | 61% average |
| Wind–solar correlation | r = −0.26 (complementary) |
| Battery cost savings (wind+solar) | 81–93% vs wind-only |
| Lower tail dependence (Clayton copula) | λ_L = 0.085 mean across 406 pairs |

---

## Repository structure

```
Renewable-Baseload/
├── Code/                              # All analysis and figure scripts
│   ├── correlation_utils.py           # Shared utilities (CV, Pearson CI, FDR)
│   │
│   ├── download_wind_data_2015_2024.py    # ENTSO-E wind data download
│   ├── download_solar_data_2015_2024.py   # ENTSO-E solar data download
│   ├── download_uk_elexon.py              # UK Elexon wind API
│   ├── download_uk_elexon_solar.py        # UK Elexon solar API
│   │
│   ├── multi_year_analysis.py             # Core multi-year trends
│   ├── monthly_analysis.py                # Monthly/seasonal statistics
│   ├── marginal_diversification.py        # Greedy portfolio construction
│   ├── dunkelflaute_analysis.py           # Wind drought event detection
│   ├── network_analysis.py                # Correlation network evolution
│   ├── tail_dependence.py                 # Clayton copula tail analysis
│   ├── combined_analysis_final.py         # Wind+solar combined analysis
│   ├── regional_clustering_combined.py    # Regional vs pan-European baseload
│   ├── battery_baseload_combined.py       # Battery storage requirements
│   ├── wavelet_coherence.py               # Wavelet time-frequency coherence
│   ├── methodological_improvements.py     # Bootstrap CI, copula GOF, CF norm
│   │
│   ├── create_figure1_improved.py         # Fig. 1
│   ├── create_figure2_diversification.py  # Fig. 2
│   ├── create_figure3_map.py              # Fig. 3 (requires cartopy)
│   └── create_figure3_seasonal.py         # Suppl. Fig. 1
│
├── Data/
│   ├── Wind/                          # 29 country CSVs (~173 MB)
│   └── Solar/                         # 28 country CSVs (~103 MB)
│
├── Results/                           # Analysis output CSVs (13 subdirectories)
├── Figures_Paper/                     # Generated figures (PDF + PNG)
│
├── CorrWindPaper_NatComm.tex          # Main paper (arXiv format)
├── CorrWindPaper_NatComm_SN.tex       # Main paper (Springer Nature format)
├── SupplementaryInformation.tex       # Supplementary Information
├── references1.bib                    # Bibliography (44 references)
│
├── run_all.py                         # Master script: reproduces everything
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # This file
```

---

## Reproducing the analysis

### Prerequisites

Python 3.9+ is required. Figure 3 (correlation map) additionally needs [cartopy](https://scitools.org.uk/cartopy/) with its system-level dependencies.

```bash
git clone https://github.com/JanKren/Renewable-Baseload.git
cd Renewable-Baseload
pip install -r requirements.txt
```

### Run everything

```bash
python3 run_all.py
```

This regenerates all Results/ CSVs and Figures_Paper/ outputs. Use `--analysis` or `--figures` to run only part of the pipeline. Individual scripts can also be run standalone:

```bash
python3 Code/multi_year_analysis.py
python3 Code/create_figure1_baseload.py
```

### Script → figure mapping

| Paper figure | Script | Also produces |
|---|---|---|
| Fig. 1 — Baseload evolution | `create_figure1_improved.py` | — |
| Fig. 2 — Diversification benefit | `create_figure2_diversification.py` | — |
| Fig. 3 — Correlation map | `create_figure3_map.py` | — |
| Fig. 4 — Scale-resolved correlation | `wavelet_coherence.py` | Results/Wavelet/ |
| Fig. 5 — Regional baseload | `regional_clustering_combined.py` | Results/Regional/ |
| Fig. 6 — Wind+solar combined | `combined_analysis_final.py` | Results/Combined/ |
| Fig. 7 — Dunkelflaute analysis | `dunkelflaute_analysis.py` | Results/Dunkelflaute/ |
| Fig. 8 — Tail dependence | `tail_dependence.py` | Results/TailDependence/ |
| Suppl. Fig. 1 — Seasonal | `create_figure3_seasonal.py` | — |
| Suppl. Fig. 2 — Marginal diversification | `marginal_diversification.py` | Results/Diversification/ |
| Suppl. Fig. 3 — Network evolution | `network_analysis.py` | Results/Network/ |
| Suppl. Fig. 4 — Battery storage | `battery_baseload_combined.py` | Results/Battery/ |
| Suppl. Fig. 5 — Greedy sensitivity | `marginal_diversification.py` | Results/Diversification/ |

---

## Data

Raw generation data are from:

- **ENTSO-E Transparency Platform** (https://transparency.entsoe.eu) — 28 countries, Actual Generation per Production Type
- **Elexon BMRS API** (https://bmrs.elexon.co.uk) — UK data, post-June 2021

Each CSV has columns: datetime index, Wind Offshore (MW), Wind Onshore (MW), Wind Total (MW) — or solar equivalents. Data span January 2015 to December 2024.

To re-download (requires API keys):

```bash
python3 Code/download_wind_data_2015_2024.py
python3 Code/download_solar_data_2015_2024.py
python3 Code/download_uk_elexon.py
python3 Code/download_uk_elexon_solar.py
```

---

## Methodological notes

- **Coefficient of variation** (CV = σ/μ) is the primary diversification metric, chosen for its dimensionless comparability across countries and years.
- **Dunkelflaute thresholds** follow Kittel & Schill (2026): severe (<10%), moderate (<20%), mild (<30%), using year-normalized thresholds (relative to each year's mean production) to separate meteorological variability from capacity growth.
- **Marginal diversification** uses greedy portfolio construction anchored on Germany (the largest producer), iteratively adding the country that minimizes portfolio CV.
- **Tail dependence** is estimated via Clayton copulas fitted to empirical uniform marginals. Lower tail λ_L measures the tendency for simultaneous extreme lows.
- All production data are in MW internally, displayed in GW in the paper.

---

## Citation

```bibtex
@article{kren2026wind,
  title   = {The wind is always blowing somewhere in {Europe}:
             {A} decade of evidence for continental-scale renewable baseload},
  author  = {Kren, Jan and Zajec, Bo{\v{s}}tjan and Tiselj, Iztok},
  journal = {Submitted to Nature Communications},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

ENTSO-E and Elexon data are subject to their respective terms of use.

## Contact

Jan Kren — jan.kren@psi.ch — [ORCID 0000-0001-8857-6167](https://orcid.org/0000-0001-8857-6167)
