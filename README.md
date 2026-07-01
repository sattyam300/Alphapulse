# AlphaPulse – Portfolio Risk & Volatility Monitor

> **Production-grade Python quantitative engine** for cross-asset portfolio risk analysis.

---

## Live Results (Last Run)

| Metric | Value |
|---|---|
| Portfolio | AAPL · TSLA · BTC-USD · GLD · JNJ |
| Weights | 20% each (equal-weight) |
| Historical Data | 1,094 daily observations (3 years) |
| Annualised Return | +16.62% |
| Annualised Volatility | 16.66% |
| **Sharpe Ratio** | **0.6976** |
| **95% VaR (USD)** | **$70,319** over 30 trading days |
| **95% VaR (%)** | **7.03%** of $1,000,000 portfolio |
| Monte Carlo Paths | 10,000 × 30 days |
| Wall-clock Time | ~1.7 seconds |

---

## Quick Start

```bash
# 1 – Install dependencies
pip install -r requirements.txt

# 2 – Run the full engine
python run_alphapulse.py

# 3 – Run tests (offline, no network needed)
pytest tests/ -v
```

---

## Project Layout

```
internship project/
├── alphapulse/
│   ├── __init__.py
│   ├── config.py            # All tunable parameters
│   ├── data_acquisition.py  # yfinance download + forward-fill
│   ├── quant_engine.py      # NumPy calculations (no loops)
│   └── output_formatter.py  # CSV / JSON serialisation
├── tests/
│   ├── conftest.py
│   └── test_alphapulse.py   # 23 offline unit tests
├── output/                  # Generated on first run
│   ├── daily_returns.csv
│   ├── rolling_volatility.csv
│   ├── correlation_matrix.json
│   ├── risk_metrics.json
│   ├── mc_distribution.csv
│   ├── dashboard_payload.json
│   └── alphapulse.log
├── run_alphapulse.py        # Entry point
└── requirements.txt
```

---

## Output Files (Tableau / Power BI ready)

| File | Purpose | Dashboard Use |
|---|---|---|
| `daily_returns.csv` | Per-asset daily log returns | Time-series line chart |
| `rolling_volatility.csv` | 30-day rolling vol per asset + portfolio | Volatility band chart |
| `correlation_matrix.json` | Symmetric matrix + long-format records | Heatmap |
| `risk_metrics.json` | VaR, Sharpe, per-asset summary stats | KPI cards |
| `mc_distribution.csv` | Histogram of 10k MC terminal values | Distribution chart |
| `dashboard_payload.json` | All-in-one REST API payload | Single-endpoint dashboard |

---

## Financial Methodology

### 1. Data & Forward Fill
Yahoo Finance adjusted close prices are downloaded for the last 3 years.
**Forward Fill** propagates the last known price into non-trading days —
the industry standard for multi-asset alignment that avoids look-ahead bias.

### 2. Daily Log Returns
```
r_t = ln(P_t / P_{t-1})
```
Log returns are time-additive and approximately normally distributed,
making them the correct input for covariance estimation and Monte Carlo.

### 3. Covariance Matrix (vectorised, no loops)
```
Σ = (1/(T-1)) · R̃ᵀ R̃ × 252
```
The demeaned return matrix `R̃` is multiplied with itself via `@` (BLAS DGEMM).
No Python-level loops. Result is annualised by ×252 trading days.

### 4. Monte Carlo Simulation (10,000 paths)
Each simulation draws correlated returns from a **Multivariate Normal**
parameterised by the historical covariance via **Cholesky decomposition**:
```
correlated_returns = Z @ Lᵀ + μ    (Z ~ N(0, I))
V_T = V_0 · exp(Σ r_t)
```
Processed in batches of 1,000 for memory efficiency.

### 5. Value at Risk (95%)
```
VaR₉₅ = V_0 − percentile(terminal_values, 5th)
```
Interpretation: with 95% confidence, the portfolio will NOT lose more than
$70,319 over the next 30 trading days.

### 6. Sharpe Ratio
```
SR = (μ_p - r_f) / σ_p   [annualised]
```
Risk-free rate: 5.0% (US 3-Month T-Bill). Sharpe of 0.70 indicates
the portfolio earns ~0.70 units of return per unit of risk taken.

### 7. 30-Day Rolling Volatility
Per-asset and portfolio rolling standard deviation, annualised by √252.
Captures market regime changes (e.g., volatility spikes during drawdowns).

---

## Configuration (`alphapulse/config.py`)

Change any parameter and re-run — no other files need editing.

| Parameter | Default | Description |
|---|---|---|
| `TICKERS` | AAPL, TSLA, BTC-USD, GLD, JNJ | Portfolio assets |
| `WEIGHTS` | [0.2, 0.2, 0.2, 0.2, 0.2] | Asset weights (auto-normalised) |
| `LOOKBACK_YEARS` | 3 | Historical data window |
| `MC_SIMULATIONS` | 10,000 | Monte Carlo iterations |
| `MC_HORIZON_DAYS` | 30 | Forecast horizon (trading days) |
| `VAR_CONFIDENCE` | 0.95 | VaR confidence level |
| `RISK_FREE_RATE_ANNUAL` | 0.05 | Risk-free rate for Sharpe |
| `ROLLING_WINDOW_DAYS` | 30 | Rolling volatility window |
