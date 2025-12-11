# Simulation Engine for Financial Time-Series

A modular Python-based simulation engine for generating multi-asset financial time-series using Monte Carlo methods, computing exposure profiles, and producing structured analytics.  
This project is intended for quantitative modeling, financial research workflows, and portfolio-level scenario analysis.

---

## Project Structure

exposure_output/  
-cva_summary.csv  
-exposure_profile.csv  
-exposure_profile.png  
-exposure_summary.json  
-simulation_diagnostics.csv  
-exposure_model.py  
-exposure_config.yaml  
-historical_prices.csv  

README.md

- **exposure_model.py** — main simulation and analytics pipeline  
- **exposure_config.yaml** — configuration for simulations, assets, and outputs  
- **historical_prices.csv** — historical price data or generated sample  
- **exposure_output/** — automatically generated reporting files  



### Multi-Asset Time-Series Simulation
- Correlated Geometric Brownian Motion (GBM)
- Risk-neutral or historical drift estimation
- Antithetic variates (optional)
- Automatic correction of non-PSD correlation matrices

### Exposure & Distribution Analytics
- Expected Exposure (EE)
- Percentile-based exposure metrics (e.g., 95th, 99th)
- Effective EPE & peak exposure detection
- Exposure term-structure reporting

### Collateral Adjustments (Optional)
- Threshold  
- Minimum Transfer Amount (MTA)  
- Margin Period of Risk (MPOR)  
- Independent Amount  

### Additional Analytics
- Diagnostics: drift error, vol error, JB normality test  
- Optional loss-based CVA-style calculations  
- Optional wrong-way stress exposure metrics  

### Reporting Outputs
- CSV exposure tables  
- JSON metric summaries  
- Diagnostics report  
- Visualization charts (PNG)

---

## Installation

Python **3.9+** recommended.

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn pyyaml tqdm
Running the Engine
1. Run with configuration
bash

python exposure_model.py --config exposure_config.yaml
This loads historical data, estimates parameters, runs simulations, computes exposures, and produces reports under exposure_output/.

2. Generate synthetic sample datasets
bash

python exposure_model.py --generate-sample --sample-output historical_prices.csv
Useful if no real price history is available.

Configuration File (exposure_config.yaml)
A typical configuration controls:

Simulation parameters

Market data inputs

Portfolio setup

Exposure percentile metrics

Output settings

Example snippet:

simulation:
  num_simulations: 5000
  horizon_days: 252
  time_steps: 252
  seed: 42
  use_antithetic: true
  confidence_level: 0.95

market_data:
  historical_prices_file: historical_prices.csv
  risk_free_rate: 0.02

positions:
  - asset_id: 1
    asset_name: "SPX"
    asset_class: "Equity"
    position_type: "Spot"
    notional: 2000000.0
    direction: 1
Output Files Explained
exposure_profile.csv
Contains:

Time grid (years / days)

Expected exposure (EE)

Confidence intervals

Percentile exposures (PFE)

Optional collateral-adjusted exposure

exposure_profile.png
Includes:

EE curve & CI band

Multiple PFE curves

Exposure term-structure plot

Optional stress overlays

simulation_diagnostics.csv
Shows:

Expected vs realized drift

Expected vs realized volatility

JB-test normality indicators

cva_summary.csv (if enabled)
Summaries of expected loss–style metrics.

exposure_summary.json
Machine-readable summary including:

Peak exposure

Effective EPE

Terminal metrics

Simulation metadata

Extensibility
You can extend the engine to:

Add stochastic volatility models

Introduce Black-Scholes or swap pricing

Run scenario analysis

Parallelize simulations

Integrate with notebooks or dashboards

Example Resume Description
Built a Python-based engine for Monte Carlo simulation of multi-asset financial time-series, generating exposure term-structures, percentile metrics, diagnostics, and automated report outputs in CSV/JSON/PNG formats.

License
This project is intended for learning, research, and portfolio demonstration.
