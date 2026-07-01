"""
AlphaPulse – Unit Test Suite
==============================
Tests validate every module independently using synthetic price data so the
suite runs offline (no network calls needed).

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from alphapulse.config import TRADING_DAYS_PER_YEAR
from alphapulse.data_acquisition import validate_weights
from alphapulse.quant_engine import (
    compute_covariance_matrix,
    compute_log_returns,
    compute_rolling_volatility,
    compute_sharpe_ratio,
    compute_var,
    monte_carlo_simulation,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """500 days of synthetic log-normal price paths for 5 assets."""
    np.random.seed(0)
    T, N = 500, 5
    tickers = ["AAPL", "TSLA", "BTC-USD", "GLD", "JNJ"]
    log_ret = np.random.randn(T, N) * 0.02 + 0.0003   # ~0.5 % daily drift
    prices = 100.0 * np.exp(log_ret.cumsum(axis=0))
    dates = pd.date_range("2021-01-04", periods=T, freq="B")
    return pd.DataFrame(prices.astype(np.float32), index=dates, columns=tickers)


@pytest.fixture
def log_returns(synthetic_prices) -> pd.DataFrame:
    return compute_log_returns(synthetic_prices)


@pytest.fixture
def equal_weights() -> np.ndarray:
    return np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)


@pytest.fixture
def cov_corr(log_returns):
    return compute_covariance_matrix(log_returns)


# ──────────────────────────────────────────────────────────────────────────────
# Data Acquisition
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateWeights:
    def test_valid_equal_weights(self):
        w = validate_weights([0.2, 0.2, 0.2, 0.2, 0.2], 5)
        assert w.shape == (5,)
        assert np.isclose(w.sum(), 1.0)

    def test_normalisation(self):
        """Weights that don't sum to 1 should be silently normalised."""
        w = validate_weights([1.0, 1.0, 1.0, 1.0, 1.0], 5)
        assert np.isclose(w.sum(), 1.0)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="Expected 5 weights"):
            validate_weights([0.5, 0.5], 5)


# ──────────────────────────────────────────────────────────────────────────────
# Log Returns
# ──────────────────────────────────────────────────────────────────────────────

class TestLogReturns:
    def test_shape(self, synthetic_prices, log_returns):
        T = len(synthetic_prices)
        N = len(synthetic_prices.columns)
        assert log_returns.shape == (T - 1, N), "Log returns should have T-1 rows"

    def test_dtype(self, log_returns):
        assert log_returns.dtypes.unique() == [np.float32]

    def test_finite(self, log_returns):
        assert np.all(np.isfinite(log_returns.values)), "No NaN/Inf in returns"

    def test_no_nans(self, log_returns):
        assert not log_returns.isnull().any().any()


# ──────────────────────────────────────────────────────────────────────────────
# Covariance & Correlation
# ──────────────────────────────────────────────────────────────────────────────

class TestCovarianceMatrix:
    def test_shape(self, cov_corr):
        cov, corr = cov_corr
        assert cov.shape == (5, 5)
        assert corr.shape == (5, 5)

    def test_symmetry(self, cov_corr):
        cov, corr = cov_corr
        np.testing.assert_allclose(cov, cov.T, atol=1e-5)
        np.testing.assert_allclose(corr, corr.T, atol=1e-5)

    def test_positive_definite(self, cov_corr):
        """All eigenvalues of the covariance matrix must be positive."""
        cov, _ = cov_corr
        eigenvalues = np.linalg.eigvalsh(cov.astype(np.float64))
        assert np.all(eigenvalues > 0), f"Non-positive eigenvalue: {eigenvalues.min()}"

    def test_correlation_diagonal_is_one(self, cov_corr):
        _, corr = cov_corr
        np.testing.assert_allclose(np.diag(corr), np.ones(5), atol=1e-5)

    def test_correlation_bounded(self, cov_corr):
        _, corr = cov_corr
        assert np.all(corr >= -1.0 - 1e-5)
        assert np.all(corr <= 1.0 + 1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Monte Carlo
# ──────────────────────────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_output_shape(self, log_returns, equal_weights, cov_corr):
        cov, _ = cov_corr
        tv = monte_carlo_simulation(log_returns, equal_weights, cov, n_simulations=500, seed=1)
        assert tv.shape == (500,)

    def test_positive_values(self, log_returns, equal_weights, cov_corr):
        """Portfolio value must always be positive (no-default assumption)."""
        cov, _ = cov_corr
        tv = monte_carlo_simulation(log_returns, equal_weights, cov, n_simulations=500, seed=2)
        assert np.all(tv > 0)

    def test_mean_near_initial_value(self, log_returns, equal_weights, cov_corr):
        """
        With synthetic data (mild random drift) over 30 days, the mean
        terminal value should be within ±5 % of the initial portfolio value.
        This tests the simulation is properly calibrated, not directionally biased.
        """
        cov, _ = cov_corr
        tv = monte_carlo_simulation(log_returns, equal_weights, cov, n_simulations=5000, seed=3)
        mean_tv = float(tv.mean())
        initial = 1_000_000.0
        assert abs(mean_tv - initial) / initial < 0.05, (
            f"Mean terminal value {mean_tv:.0f} deviates more than 5% from {initial:.0f}"
        )

    def test_reproducibility(self, log_returns, equal_weights, cov_corr):
        cov, _ = cov_corr
        tv1 = monte_carlo_simulation(log_returns, equal_weights, cov, n_simulations=200, seed=99)
        tv2 = monte_carlo_simulation(log_returns, equal_weights, cov, n_simulations=200, seed=99)
        np.testing.assert_array_equal(tv1, tv2)


# ──────────────────────────────────────────────────────────────────────────────
# VaR
# ──────────────────────────────────────────────────────────────────────────────

class TestVaR:
    def test_structure(self, log_returns, equal_weights, cov_corr):
        cov, _ = cov_corr
        tv = monte_carlo_simulation(log_returns, equal_weights, cov, n_simulations=500, seed=4)
        result = compute_var(tv)
        assert "var_dollar" in result
        assert "var_pct" in result
        assert result["var_dollar"] > 0, "VaR must be a positive loss figure"
        assert 0 < result["var_pct"] < 100

    def test_var_below_100_pct(self, log_returns, equal_weights, cov_corr):
        cov, _ = cov_corr
        tv = monte_carlo_simulation(log_returns, equal_weights, cov, n_simulations=500, seed=5)
        result = compute_var(tv)
        assert result["var_pct"] < 100, "Can't lose more than 100% of initial capital"


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe Ratio
# ──────────────────────────────────────────────────────────────────────────────

class TestSharpe:
    def test_keys_present(self, log_returns, equal_weights):
        result = compute_sharpe_ratio(log_returns, equal_weights)
        assert {"sharpe_ratio", "annualised_return", "annualised_volatility"} <= result.keys()

    def test_volatility_positive(self, log_returns, equal_weights):
        result = compute_sharpe_ratio(log_returns, equal_weights)
        assert result["annualised_volatility"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# Rolling Volatility
# ──────────────────────────────────────────────────────────────────────────────

class TestRollingVolatility:
    def test_output_has_portfolio_column(self, log_returns, equal_weights):
        rv = compute_rolling_volatility(log_returns, equal_weights, window=20)
        assert "Portfolio" in rv.columns

    def test_first_rows_are_nan(self, log_returns, equal_weights):
        window = 20
        rv = compute_rolling_volatility(log_returns, equal_weights, window=window)
        assert rv.iloc[:window - 1].isnull().all().all(), \
            "First window-1 rows must be NaN"

    def test_values_positive_after_warmup(self, log_returns, equal_weights):
        window = 20
        rv = compute_rolling_volatility(log_returns, equal_weights, window=window)
        valid = rv.iloc[window:].dropna()
        assert (valid > 0).all().all(), "All non-NaN rolling vol values must be > 0"
