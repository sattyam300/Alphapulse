"""
AlphaPulse – Quantitative Engine
=================================
This module is the analytical core of AlphaPulse.  Every heavy calculation
is vectorised via NumPy to avoid Python-level loops and exploit BLAS/LAPACK
routines under the hood.

Financial Methodology
---------------------
Log Returns
~~~~~~~~~~~
    r_t = ln(P_t / P_{t-1})

Log returns are preferred over simple returns for multi-period aggregation
because they are time-additive and approximately normally distributed —
a key assumption of many risk models.

Covariance Matrix
~~~~~~~~~~~~~~~~~
Given the (T × N) demeaned return matrix R̃, the sample covariance matrix is:

    Σ = (1 / (T-1)) * R̃ᵀ R̃

Using ``np.dot`` / ``@`` operator avoids explicit Python loops.  We then
annualise by multiplying by 252 (trading days per year).

Monte Carlo Portfolio Value Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each of the 10 000+ simulations draws a T×N matrix of correlated daily
returns from a multivariate normal distribution parameterised by the
estimated covariance matrix.  The portfolio is assumed to start at
$1 000 000 and compound daily over the chosen horizon.

Value at Risk (VaR) – Parametric & Historical
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
95 % VaR = the loss at the 5th percentile of simulated terminal portfolio
values.  Reported as a positive dollar figure (i.e. "at 95 % confidence,
you will NOT lose more than $X over the next 30 trading days").

Sharpe Ratio (annualised)
~~~~~~~~~~~~~~~~~~~~~~~~~
    SR = (μ_p - r_f) / σ_p

where μ_p and σ_p are the annualised portfolio return and volatility, and
r_f is the risk-free rate.

Rolling Volatility
~~~~~~~~~~~~~~~~~~
A 30-day rolling standard deviation of daily log returns, annualised by
√252.  This captures regime changes in market uncertainty.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from alphapulse.config import (
    MC_HORIZON_DAYS,
    MC_SIMULATIONS,
    RISK_FREE_RATE_ANNUAL,
    ROLLING_WINDOW_DAYS,
    TRADING_DAYS_PER_YEAR,
    VAR_CONFIDENCE,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  LOG RETURNS
# ──────────────────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from adjusted close prices.

    Formula: r_t = ln(P_t / P_{t-1})

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed price DataFrame, shape (T, N).

    Returns
    -------
    pd.DataFrame
        Log-return DataFrame, shape (T-1, N), float32.
    """
    logger.info("Computing daily log returns …")

    # np.log(prices / prices.shift(1)) — fully vectorised, no Python loops.
    # We use the underlying NumPy array for the division to avoid pandas
    # alignment overhead, then re-attach the index and columns.
    price_arr: np.ndarray = prices.values.astype(np.float32)

    # log(P_t) - log(P_{t-1})  ←→  log(P_t / P_{t-1})
    log_ret_arr: np.ndarray = np.log(price_arr[1:] / price_arr[:-1]).astype(np.float32)

    log_returns = pd.DataFrame(
        log_ret_arr,
        index=prices.index[1:],
        columns=prices.columns,
    )

    logger.info("Log returns shape: %s", log_returns.shape)
    return log_returns


# ──────────────────────────────────────────────────────────────────────────────
# 2.  COVARIANCE & CORRELATION MATRICES
# ──────────────────────────────────────────────────────────────────────────────

def compute_covariance_matrix(
    log_returns: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the **annualised** sample covariance matrix via vectorised
    matrix multiplication (no explicit loops).

    Methodology
    -----------
    1. Extract the (T × N) NumPy array R.
    2. Demean each column: R̃ = R - mean(R, axis=0).
    3. Cov = (1/(T-1)) * R̃ᵀ @ R̃   ← single matrix multiply.
    4. Annualise: Σ_annual = Cov * 252.
    5. Derive Pearson correlation: diag(σ)^{-1} Σ diag(σ)^{-1}.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return DataFrame, shape (T, N).

    Returns
    -------
    cov_annual : np.ndarray
        Annualised covariance matrix, shape (N, N), float32.
    corr_matrix : np.ndarray
        Correlation matrix, shape (N, N), float32.
    """
    logger.info("Computing covariance matrix via vectorised matrix multiplication …")

    R: np.ndarray = log_returns.values.astype(np.float64)   # float64 for precision
    T, N = R.shape

    # Step 1 – Demean
    R_demeaned: np.ndarray = R - R.mean(axis=0)

    # Step 2 – Sample covariance via dot product (BLAS DGEMM under the hood)
    #   Σ = (1/(T-1)) * R̃ᵀ R̃
    cov_daily: np.ndarray = (R_demeaned.T @ R_demeaned) / (T - 1)

    # Step 3 – Annualise
    cov_annual: np.ndarray = (cov_daily * TRADING_DAYS_PER_YEAR).astype(np.float32)

    # Step 4 – Derive Pearson correlation matrix
    std_vec: np.ndarray = np.sqrt(np.diag(cov_annual))          # shape (N,)
    outer_std: np.ndarray = np.outer(std_vec, std_vec)           # shape (N, N)
    corr_matrix: np.ndarray = (cov_annual / outer_std).astype(np.float32)

    logger.info("Covariance & correlation matrices computed ✓")
    return cov_annual, corr_matrix


# ──────────────────────────────────────────────────────────────────────────────
# 3.  MONTE CARLO SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

def monte_carlo_simulation(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    cov_annual: np.ndarray,
    *,
    initial_value: float = 1_000_000.0,
    n_simulations: int = MC_SIMULATIONS,
    horizon_days: int = MC_HORIZON_DAYS,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate ``n_simulations`` portfolio value paths over ``horizon_days``.

    Each simulation draws correlated daily log-returns from a multivariate
    normal distribution calibrated to the historical covariance matrix, then
    compounds them into a terminal portfolio value.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Historical log returns (used for mean estimation).
    weights : np.ndarray
        Portfolio weight vector, shape (N,).
    cov_annual : np.ndarray
        Annualised covariance matrix, shape (N, N).
    initial_value : float
        Starting portfolio value in USD.
    n_simulations : int
        Number of Monte Carlo paths (≥ 10 000 for convergence).
    horizon_days : int
        Forecast horizon in trading days.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    terminal_values : np.ndarray
        Array of simulated terminal portfolio values, shape (n_simulations,).
        float32.
    """
    logger.info(
        "Running Monte Carlo: %d simulations × %d days …",
        n_simulations,
        horizon_days,
    )

    rng = np.random.default_rng(seed)

    # Daily mean log-returns per asset (from historical data)
    daily_mean: np.ndarray = log_returns.values.mean(axis=0).astype(np.float64)   # (N,)

    # Daily covariance (de-annualise)
    cov_daily: np.ndarray = (cov_annual / TRADING_DAYS_PER_YEAR).astype(np.float64)

    # Cholesky decomposition: Σ = L Lᵀ
    # Drawing Z ~ N(0, I) then transforming by L gives correlated returns.
    L: np.ndarray = np.linalg.cholesky(cov_daily)   # (N, N)

    N = len(daily_mean)

    # Pre-allocate terminal values array
    terminal_values: np.ndarray = np.empty(n_simulations, dtype=np.float32)

    # ── Batched simulation: process BATCH paths at a time to balance ──────
    # memory use and vectorisation efficiency.
    #
    # Tensor layout chosen for clarity and correctness:
    #   Z           : (batch_size, horizon_days, N)   — iid standard normals
    #   correlated  : (batch_size, horizon_days, N)   — after L @ Z
    #   daily_rets  : (batch_size, horizon_days, N)   — + drift
    #   port_log_ret: (batch_size, horizon_days)       — dot with weights
    BATCH = 1_000
    w_f64 = weights.astype(np.float64)  # (N,)

    for batch_start in range(0, n_simulations, BATCH):
        batch_end = min(batch_start + BATCH, n_simulations)
        batch_size = batch_end - batch_start

        # Step 1 – Independent standard normal shocks
        # shape: (batch_size, horizon_days, N)
        Z: np.ndarray = rng.standard_normal((batch_size, horizon_days, N))

        # Step 2 – Introduce cross-asset correlation via Cholesky factor
        # Z @ Lᵀ yields correlated returns because Cov(L z) = L I Lᵀ = Σ
        # shape remains (batch_size, horizon_days, N)
        correlated: np.ndarray = Z @ L.T   # (batch, horizon, N) @ (N, N)

        # Step 3 – Add mean drift to every day in every simulation
        daily_rets: np.ndarray = correlated + daily_mean  # broadcasts over (batch, horizon)

        # Step 4 – Weighted portfolio log-return per day
        # (batch_size, horizon_days, N) @ (N,) → (batch_size, horizon_days)
        port_log_ret: np.ndarray = daily_rets @ w_f64

        # Step 5 – Compound over the full horizon: V_T = V_0 · exp(Σ_t r_t)
        cumulative_log_ret: np.ndarray = port_log_ret.sum(axis=1)  # (batch_size,)
        terminal_values[batch_start:batch_end] = (
            initial_value * np.exp(cumulative_log_ret)
        ).astype(np.float32)

    logger.info("Monte Carlo simulation complete ✓")
    return terminal_values


# ──────────────────────────────────────────────────────────────────────────────
# 4.  RISK METRICS: VaR & SHARPE RATIO
# ──────────────────────────────────────────────────────────────────────────────

def compute_var(
    terminal_values: np.ndarray,
    initial_value: float = 1_000_000.0,
    confidence: float = VAR_CONFIDENCE,
) -> dict:
    """
    Compute Value at Risk (VaR) from Monte Carlo terminal portfolio values.

    VaR is defined as the loss (positive number) that is NOT exceeded with
    probability ``confidence`` over the simulation horizon.

    Parameters
    ----------
    terminal_values : np.ndarray
        Simulated terminal portfolio values, shape (n_simulations,).
    initial_value : float
        Starting portfolio value.
    confidence : float
        Confidence level (e.g. 0.95 for 95 % VaR).

    Returns
    -------
    dict
        ``var_dollar``  : VaR in USD (positive = potential loss).
        ``var_pct``     : VaR as a percentage of initial value.
        ``percentile``  : The terminal value at the (1-confidence) percentile.
    """
    percentile_value: float = float(np.percentile(terminal_values, (1 - confidence) * 100))
    var_dollar: float = float(initial_value - percentile_value)
    var_pct: float = var_dollar / initial_value * 100

    logger.info(
        "95%% VaR = $%.2f  (%.2f%% of portfolio)",
        var_dollar, var_pct,
    )
    return {
        "var_dollar": round(var_dollar, 2),
        "var_pct": round(var_pct, 4),
        "percentile_value": round(percentile_value, 2),
        "confidence_level": confidence,
        "initial_value": initial_value,
    }


def compute_sharpe_ratio(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = RISK_FREE_RATE_ANNUAL,
) -> dict:
    """
    Compute the annualised Sharpe Ratio for the portfolio.

    Formula: SR = (μ_p - r_f) / σ_p
    where μ_p and σ_p are annualised portfolio return and volatility.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return DataFrame, shape (T, N).
    weights : np.ndarray
        Portfolio weight vector, shape (N,).
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.05 for 5 %).

    Returns
    -------
    dict
        ``sharpe_ratio``, ``annualised_return``, ``annualised_volatility``.
    """
    # Daily portfolio return = weighted sum of asset log returns
    daily_port_ret: np.ndarray = log_returns.values.astype(np.float32) @ weights

    # Annualise
    mean_daily: float = float(daily_port_ret.mean())
    std_daily: float = float(daily_port_ret.std(ddof=1))

    annualised_return: float = mean_daily * TRADING_DAYS_PER_YEAR
    annualised_vol: float = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    daily_rf: float = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_daily: float = float((daily_port_ret - daily_rf).mean())
    sharpe: float = excess_daily * np.sqrt(TRADING_DAYS_PER_YEAR) / std_daily

    logger.info(
        "Sharpe Ratio = %.4f  |  Ann. Return = %.4f  |  Ann. Vol = %.4f",
        sharpe, annualised_return, annualised_vol,
    )
    return {
        "sharpe_ratio": round(float(sharpe), 4),
        "annualised_return": round(annualised_return, 6),
        "annualised_volatility": round(annualised_vol, 6),
        "risk_free_rate": risk_free_rate,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  ROLLING VOLATILITY
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_volatility(
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    window: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Compute 30-day rolling volatility (annualised) for each asset AND
    for the weighted portfolio.

    Rolling Std is computed per-column using pandas' optimised Cython
    implementation, then multiplied by √252 to annualise.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns, shape (T, N).
    weights : np.ndarray
        Portfolio weight vector, shape (N,).
    window : int
        Rolling window length in trading days.

    Returns
    -------
    pd.DataFrame
        Annualised rolling volatility for each asset plus a ``"Portfolio"``
        column, shape (T, N+1). First (window-1) rows are NaN by design.
    """
    logger.info("Computing %d-day rolling volatility …", window)

    # Per-asset rolling std (annualised)
    roll_vol: pd.DataFrame = (
        log_returns
        .rolling(window=window, min_periods=window)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).astype(np.float32)

    # Portfolio daily log return, then rolling std
    port_daily_ret: pd.Series = pd.Series(
        log_returns.values.astype(np.float32) @ weights,
        index=log_returns.index,
        name="Portfolio",
    )
    port_roll_vol: pd.Series = (
        port_daily_ret
        .rolling(window=window, min_periods=window)
        .std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    ).astype(np.float32)

    result = pd.concat([roll_vol, port_roll_vol], axis=1)

    logger.info("Rolling volatility computed ✓")
    return result
