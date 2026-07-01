"""
AlphaPulse – Main Orchestrator
================================
Entry point for the AlphaPulse Portfolio Risk & Volatility Monitor.

Usage
-----
    python run_alphapulse.py

The script orchestrates all modules in sequence:
1. Data Acquisition   (alphapulse.data_acquisition)
2. Quantitative Engine (alphapulse.quant_engine)
3. Output Formatter   (alphapulse.output_formatter)

All output files land in the ``output/`` directory.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

from alphapulse.config import (
    MC_HORIZON_DAYS,
    MC_SIMULATIONS,
    RISK_FREE_RATE_ANNUAL,
    ROLLING_WINDOW_DAYS,
    TICKERS,
    WEIGHTS,
    VAR_CONFIDENCE,
)
from alphapulse.data_acquisition import fetch_prices, validate_weights
from alphapulse.quant_engine import (
    compute_covariance_matrix,
    compute_log_returns,
    compute_rolling_volatility,
    compute_sharpe_ratio,
    compute_var,
    monte_carlo_simulation,
)
from alphapulse.output_formatter import (
    save_correlation_json,
    save_dashboard_payload,
    save_mc_distribution_csv,
    save_returns_csv,
    save_risk_metrics_json,
    save_rolling_vol_csv,
    save_tableau_csvs,
    save_tableau_excel,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────

# Create output directory FIRST so the FileHandler can open the log file.
Path("output").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("output/alphapulse.log", mode="w"),
    ],
)
logger = logging.getLogger("alphapulse.main")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.perf_counter()

    logger.info("=" * 68)
    logger.info("  AlphaPulse – Portfolio Risk & Volatility Monitor  ")
    logger.info("=" * 68)
    logger.info("Portfolio : %s", TICKERS)
    logger.info("Weights   : %s", WEIGHTS)
    logger.info("MC Paths  : %d  |  Horizon: %d days", MC_SIMULATIONS, MC_HORIZON_DAYS)
    logger.info("VaR Conf. : %.0f%%  |  RF Rate: %.2f%%", VAR_CONFIDENCE * 100, RISK_FREE_RATE_ANNUAL * 100)

    # ── STEP 1 : Data Acquisition ─────────────────────────────────────────
    logger.info("\n── STEP 1/5  Data Acquisition ──────────────────────────────────")
    prices = fetch_prices()
    weights: np.ndarray = validate_weights(WEIGHTS, len(prices.columns))

    # ── STEP 2 : Quantitative Calculations ────────────────────────────────
    logger.info("\n── STEP 2/5  Quantitative Engine ───────────────────────────────")
    log_returns = compute_log_returns(prices)
    cov_annual, corr_matrix = compute_covariance_matrix(log_returns)

    # ── STEP 3 : Monte Carlo + Risk Metrics ───────────────────────────────
    logger.info("\n── STEP 3/5  Monte Carlo Simulation ────────────────────────────")
    terminal_values = monte_carlo_simulation(
        log_returns=log_returns,
        weights=weights,
        cov_annual=cov_annual,
    )
    var_metrics  = compute_var(terminal_values)
    sharpe_metrics = compute_sharpe_ratio(log_returns, weights)

    # ── STEP 4 : Rolling Volatility ───────────────────────────────────────
    logger.info("\n── STEP 4/5  Rolling Volatility ─────────────────────────────────")
    rolling_vol = compute_rolling_volatility(log_returns, weights, window=ROLLING_WINDOW_DAYS)

    # ── STEP 5 : Output Serialisation ────────────────────────────────────
    logger.info("\n── STEP 5/5  Serialising Output ─────────────────────────────────")
    save_returns_csv(log_returns)
    save_rolling_vol_csv(rolling_vol)
    save_correlation_json(corr_matrix)
    save_risk_metrics_json(var_metrics, sharpe_metrics, log_returns, weights)
    save_mc_distribution_csv(terminal_values)
    save_dashboard_payload(
        log_returns=log_returns,
        rolling_vol=rolling_vol,
        corr_matrix=corr_matrix,
        var_metrics=var_metrics,
        sharpe_metrics=sharpe_metrics,
        weights=weights,
    )

    # ── Tableau flat CSV bundle ───────────────────────────────────────────
    logger.info("\n── Tableau Export ────────────────────────────────────────────────")
    save_tableau_csvs(
        corr_matrix=corr_matrix,
        var_metrics=var_metrics,
        sharpe_metrics=sharpe_metrics,
        log_returns=log_returns,
        rolling_vol=rolling_vol,
        terminal_values=terminal_values,
        weights=weights,
    )

    # ── Excel workbook for Tableau Public ────────────────────────────────
    logger.info("\n── Excel Export (Tableau Public) ─────────────────────────────────")
    save_tableau_excel(
        corr_matrix=corr_matrix,
        var_metrics=var_metrics,
        sharpe_metrics=sharpe_metrics,
        log_returns=log_returns,
        rolling_vol=rolling_vol,
        terminal_values=terminal_values,
        weights=weights,
    )

    elapsed = time.perf_counter() - t0

    logger.info("\n" + "=" * 68)
    logger.info("  AlphaPulse Complete  (%.2f s)", elapsed)
    logger.info("=" * 68)

    # ── Print final summary to stdout ─────────────────────────────────────
    print("\n" + "╔" + "═" * 66 + "╗")
    print("║{:^66}║".format(" AlphaPulse – Summary Report "))
    print("╠" + "═" * 66 + "╣")
    print(f"║  {'Portfolio':<30} {', '.join(TICKERS):<35}║")
    print(f"║  {'Weights':<30} {str([round(float(w), 2) for w in weights]):<35}║")
    print(f"║  {'Data Points':<30} {len(log_returns):<35}║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  {'Annualised Return':<30} {sharpe_metrics['annualised_return']:>+.4%}{'':>21}║")
    print(f"║  {'Annualised Volatility':<30} {sharpe_metrics['annualised_volatility']:>.4%}{'':>21}║")
    print(f"║  {'Sharpe Ratio':<30} {sharpe_metrics['sharpe_ratio']:>+.4f}{'':>25}║")
    print(f"║  {'95% VaR (USD)':<30} ${var_metrics['var_dollar']:>,.2f}{'':>22}║")
    print(f"║  {'95% VaR (%)':<30} {var_metrics['var_pct']:.2f}%{'':>31}║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  {'Output files in':<30} ./output/{'':>26}║")
    print(f"║  {'Wall-clock time':<30} {elapsed:.2f}s{'':>31}║")
    print("╚" + "═" * 66 + "╝\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Run interrupted by user.")
        sys.exit(0)
