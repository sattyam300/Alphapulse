"""
AlphaPulse Configuration
========================
Central place for all tunable parameters so the engine remains
easily maintainable without touching business-logic code.
"""

from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Portfolio Definition
# ---------------------------------------------------------------------------
# Tickers span equities (tech + healthcare), crypto, and commodities —
# giving meaningful cross-asset correlation signals in the output heatmap.
TICKERS: list[str] = ["AAPL", "TSLA", "BTC-USD", "GLD", "JNJ"]

# Equal-weight portfolio (must sum to 1.0)
WEIGHTS: list[float] = [0.20, 0.20, 0.20, 0.20, 0.20]

# ---------------------------------------------------------------------------
# Data Window
# ---------------------------------------------------------------------------
LOOKBACK_YEARS: int = 3
END_DATE: str = date.today().isoformat()
START_DATE: str = (date.today() - timedelta(days=365 * LOOKBACK_YEARS)).isoformat()

# ---------------------------------------------------------------------------
# Risk-Free Rate (annualised, e.g. 5 % US 3-Month T-Bill)
# ---------------------------------------------------------------------------
RISK_FREE_RATE_ANNUAL: float = 0.05
TRADING_DAYS_PER_YEAR: int = 252

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
MC_SIMULATIONS: int = 10_000    # number of portfolio paths
MC_HORIZON_DAYS: int = 30       # forecast horizon in trading days

# ---------------------------------------------------------------------------
# Rolling Window
# ---------------------------------------------------------------------------
ROLLING_WINDOW_DAYS: int = 30   # 30-day rolling volatility

# ---------------------------------------------------------------------------
# VaR Confidence Level
# ---------------------------------------------------------------------------
VAR_CONFIDENCE: float = 0.95    # 95 % VaR

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = "output"
RETURNS_CSV: str = f"{OUTPUT_DIR}/daily_returns.csv"
ROLLING_VOL_CSV: str = f"{OUTPUT_DIR}/rolling_volatility.csv"
CORRELATION_JSON: str = f"{OUTPUT_DIR}/correlation_matrix.json"
RISK_METRICS_JSON: str = f"{OUTPUT_DIR}/risk_metrics.json"
MC_DISTRIBUTION_CSV: str = f"{OUTPUT_DIR}/mc_distribution.csv"
DASHBOARD_JSON: str = f"{OUTPUT_DIR}/dashboard_payload.json"
