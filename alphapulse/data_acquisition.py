"""
AlphaPulse – Data Acquisition Module
=====================================
Responsibilities
----------------
1. Download adjusted close prices from Yahoo Finance via yfinance.
2. Apply a "Forward Fill" (ffill) strategy for weekends / holidays where
   some assets (e.g. BTC-USD) trade but others do not.  ffill propagates
   the last known price forward, preserving temporal continuity without
   introducing look-ahead bias.
3. Return a tidy, date-indexed DataFrame with float32 precision to
   minimise memory footprint during downstream NumPy calculations.

Financial Note on Forward Fill
-------------------------------
Forward fill is the industry-standard approach for multi-asset data
alignment.  Back-fill (bfill) would introduce look-ahead bias and is
explicitly avoided.  Any leading NaNs (assets listed after the start date)
are dropped because no prior price exists to propagate.

Cache Note
----------
On macOS the default yfinance SQLite cache lives in ~/Library/Caches/py-yfinance.
If another process holds a lock on that DB the download silently fails.
We redirect the cache to .yf_cache/ inside the project folder to avoid this.
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Redirect yfinance cache to a project-local directory so the macOS
# system cache (often locked by other processes) does not interfere.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".yf_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
try:
    yf.set_tz_cache_location(_CACHE_DIR)
except Exception:
    pass  # Silently ignore if the API is unavailable in this yfinance version

from alphapulse.config import (
    END_DATE,
    START_DATE,
    TICKERS,
)

logger = logging.getLogger(__name__)


def fetch_prices(
    tickers: list[str] = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    *,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Download 3-year daily adjusted close prices for *tickers*.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols.
    start : str
        ISO-8601 start date, e.g. ``"2021-04-18"``.
    end : str
        ISO-8601 end date (exclusive in yfinance).
    progress : bool
        Show yfinance download progress bar (disabled by default for
        cleaner log output in production).

    Returns
    -------
    pd.DataFrame
        Shape ``(trading_days, n_assets)``.  Columns are ticker symbols.
        Values are adjusted close prices in float32.

    Raises
    ------
    ValueError
        If the resulting DataFrame is empty or has fewer than 2 tickers
        with usable data.
    """
    logger.info("Fetching price data: tickers=%s  start=%s  end=%s", tickers, start, end)

    # Retry loop: handles transient network errors and SQLite cache locks.
    last_exc: Exception | None = None
    raw: pd.DataFrame = pd.DataFrame()
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,   # returns adjusted close directly in "Close"
                progress=progress,
                threads=False,      # serial to avoid SQLite write conflicts
            )
            if not raw.empty:
                break
        except Exception as exc:
            last_exc = exc
            logger.warning("Download attempt %d failed: %s — retrying in 3 s …", attempt, exc)
            time.sleep(3)

    if raw.empty:
        detail = str(last_exc) if last_exc else "Empty response from Yahoo Finance"
        raise RuntimeError(f"yfinance download failed after 3 attempts: {detail}")

    # ── Extract "Close" level when yfinance returns a MultiIndex ──────────
    if isinstance(raw.columns, pd.MultiIndex):
        prices: pd.DataFrame = raw["Close"].copy()
    else:
        # Single-ticker edge-case (unlikely here, but defensive)
        prices = raw.copy()

    # Ensure column names match the requested ticker list (preserving order)
    prices = prices[tickers]

    # ── Forward Fill: propagate last valid price into non-trading days ────
    # This is applied BEFORE dropping rows so that leading rows with ALL
    # NaNs are preserved temporarily for the drop step below.
    prices = prices.ffill()

    # ── Drop rows where ALL assets are still NaN (true market closures) ──
    prices.dropna(how="all", inplace=True)

    # ── Drop any leading rows still containing NaN (e.g. assets that      ──
    #    had no prior price to forward-fill from, like BTC-USD early data) ──
    prices.dropna(inplace=True)

    if prices.empty:
        raise ValueError(
            "Price data is empty after cleaning. "
            "Check ticker symbols and date range."
        )

    if prices.shape[1] < 2:
        raise ValueError(
            f"Only {prices.shape[1]} ticker(s) with usable data. "
            "Need at least 2 for covariance / correlation analysis."
        )

    logger.info(
        "Price data ready: %d trading days × %d assets.",
        prices.shape[0],
        prices.shape[1],
    )

    # ── Cast to float32 for memory efficiency ────────────────────────────
    prices = prices.astype(np.float32)

    return prices


def validate_weights(weights: list[float], n_assets: int) -> np.ndarray:
    """
    Validate and normalise portfolio weights.

    Parameters
    ----------
    weights : list[float]
        User-provided weights (should sum to 1.0).
    n_assets : int
        Number of assets in the portfolio.

    Returns
    -------
    np.ndarray
        Normalised weight vector as float32.

    Raises
    ------
    ValueError
        If the number of weights does not match ``n_assets``.
    """
    if len(weights) != n_assets:
        raise ValueError(
            f"Expected {n_assets} weights, got {len(weights)}."
        )

    w = np.array(weights, dtype=np.float32)
    total = w.sum()

    if not np.isclose(total, 1.0, atol=1e-4):
        logger.warning(
            "Weights sum to %.6f; normalising to 1.0 automatically.", total
        )
        w /= total

    return w
