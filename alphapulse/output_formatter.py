import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from alphapulse.config import (
    CORRELATION_JSON,
    DASHBOARD_JSON,
    MC_DISTRIBUTION_CSV,
    MC_SIMULATIONS,
    OUTPUT_DIR,
    RETURNS_CSV,
    RISK_METRICS_JSON,
    ROLLING_VOL_CSV,
    TICKERS,
)

logger = logging.getLogger(__name__)


def _ensure_output_dir() -> None:
    """Create output directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_float(value: Any) -> float:
    """Cast numpy scalars / arrays to plain Python float for JSON safety."""
    return float(value)


# ──────────────────────────────────────────────────────────────────────────────
# Individual serialisers
# ──────────────────────────────────────────────────────────────────────────────

def save_returns_csv(log_returns: pd.DataFrame) -> str:
    """Write daily log returns to CSV.  Index column is named 'Date'."""
    _ensure_output_dir()
    log_returns.index.name = "Date"
    log_returns.to_csv(RETURNS_CSV, float_format="%.8f")
    logger.info("Saved → %s", RETURNS_CSV)
    return RETURNS_CSV


def save_rolling_vol_csv(rolling_vol: pd.DataFrame) -> str:
    """Write 30-day rolling volatility to CSV."""
    _ensure_output_dir()
    rolling_vol.index.name = "Date"
    rolling_vol.to_csv(ROLLING_VOL_CSV, float_format="%.8f")
    logger.info("Saved → %s", ROLLING_VOL_CSV)
    return ROLLING_VOL_CSV


def save_correlation_json(
    corr_matrix: np.ndarray,
    tickers: list[str] = TICKERS,
) -> str:
    """
    Write the correlation matrix as a structured JSON object suitable for a
    heatmap visual in Tableau / Power BI.

    Schema
    ------
    .. code-block:: json

        {
          "tickers": ["AAPL", "TSLA", ...],
          "matrix": [[1.0, 0.43, ...], ...],
          "records": [{"asset_x": "AAPL", "asset_y": "TSLA", "correlation": 0.43}, ...]
        }

    The ``records`` list is in tidy (long) format — directly importable as a
    table in Tableau / Power BI with no further transformation needed.
    """
    _ensure_output_dir()

    # Tidy / long format for heatmap
    records = []
    n = len(tickers)
    for i in range(n):
        for j in range(n):
            records.append({
                "asset_x": tickers[i],
                "asset_y": tickers[j],
                "correlation": round(_safe_float(corr_matrix[i, j]), 6),
            })

    payload = {
        "tickers": tickers,
        "matrix": [
            [round(_safe_float(corr_matrix[i, j]), 6) for j in range(n)]
            for i in range(n)
        ],
        "records": records,
    }

    with open(CORRELATION_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Saved → %s", CORRELATION_JSON)
    return CORRELATION_JSON


def save_risk_metrics_json(
    var_metrics: dict,
    sharpe_metrics: dict,
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    tickers: list[str] = TICKERS,
) -> str:
    """
    Aggregate all scalar risk metrics into a single JSON file.

    Includes per-asset summary stats (mean, std, min, max) for the
    dashboard overview cards.
    """
    _ensure_output_dir()

    # Per-asset summary statistics
    asset_stats = {}
    for ticker in tickers:
        col = log_returns[ticker].dropna()
        asset_stats[ticker] = {
            "mean_daily_return": round(_safe_float(col.mean()), 8),
            "daily_volatility":  round(_safe_float(col.std(ddof=1)), 8),
            "annual_volatility": round(_safe_float(col.std(ddof=1) * np.sqrt(252)), 6),
            "min_return":        round(_safe_float(col.min()), 8),
            "max_return":        round(_safe_float(col.max()), 8),
            "skewness":          round(_safe_float(float(col.skew())), 6),
            "kurtosis":          round(_safe_float(float(col.kurt())), 6),
        }

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "portfolio": {
            "tickers": tickers,
            "weights": [round(_safe_float(w), 6) for w in weights],
        },
        "var_95": var_metrics,
        "sharpe": sharpe_metrics,
        "asset_statistics": asset_stats,
    }

    with open(RISK_METRICS_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Saved → %s", RISK_METRICS_JSON)
    return RISK_METRICS_JSON


def save_mc_distribution_csv(
    terminal_values: np.ndarray,
    n_bins: int = 100,
) -> str:
    """
    Bin Monte Carlo terminal values into a histogram and save as CSV.

    Power BI / Tableau can use ``bin_midpoint`` (x-axis) and
    ``frequency`` (y-axis) to render a distribution chart directly.

    Parameters
    ----------
    terminal_values : np.ndarray
        Simulated terminal portfolio values, shape (n_simulations,).
    n_bins : int
        Number of histogram bins.
    """
    _ensure_output_dir()

    counts, bin_edges = np.histogram(terminal_values, bins=n_bins)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    df = pd.DataFrame({
        "bin_midpoint_usd": bin_midpoints.astype(np.float32),
        "frequency":        counts,
        "probability":      (counts / counts.sum()).astype(np.float32),
    })
    df.to_csv(MC_DISTRIBUTION_CSV, index=False, float_format="%.4f")

    logger.info("Saved → %s  (%d bins over %d simulations)", MC_DISTRIBUTION_CSV, n_bins, MC_SIMULATIONS)
    return MC_DISTRIBUTION_CSV


def save_dashboard_payload(
    log_returns: pd.DataFrame,
    rolling_vol: pd.DataFrame,
    corr_matrix: np.ndarray,
    var_metrics: dict,
    sharpe_metrics: dict,
    weights: np.ndarray,
    tickers: list[str] = TICKERS,
) -> str:
    """
    Consolidate ALL outputs into a single ``dashboard_payload.json``.

    Useful for REST API backends that serve dashboards in one request.

    Schema sections
    ---------------
    - ``meta``           : run timestamp, tickers, weights
    - ``risk_metrics``   : VaR, Sharpe
    - ``correlation``    : heatmap records (long format)
    - ``returns_tail``   : last 60 rows of log returns (preview)
    - ``rolling_vol_tail``: last 60 rows of rolling volatility (preview)
    """
    _ensure_output_dir()

    n = len(tickers)
    corr_records = [
        {
            "asset_x": tickers[i],
            "asset_y": tickers[j],
            "correlation": round(_safe_float(corr_matrix[i, j]), 6),
        }
        for i in range(n) for j in range(n)
    ]

    # Convert tail DataFrames to list-of-dicts (JSON-serialisable)
    returns_tail = (
        log_returns.tail(60)
        .reset_index()
        .rename(columns={"Date": "date"})
        .assign(date=lambda df: df["date"].astype(str))
        .map(lambda x: round(float(x), 8) if isinstance(x, (float, np.floating)) else x)
        .to_dict(orient="records")
    )

    rolling_tail = (
        rolling_vol.tail(60)
        .reset_index()
        .rename(columns={"Date": "date"})
        .assign(date=lambda df: df["date"].astype(str))
        .map(lambda x: round(float(x), 8) if isinstance(x, (float, np.floating)) else x)
        .to_dict(orient="records")
    )

    payload = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "tickers": tickers,
            "weights": [round(_safe_float(w), 6) for w in weights],
            "data_points_total": len(log_returns),
        },
        "risk_metrics": {
            "var_95": var_metrics,
            "sharpe": sharpe_metrics,
        },
        "correlation_heatmap": corr_records,
        "returns_preview_last60": returns_tail,
        "rolling_vol_preview_last60": rolling_tail,
    }

    with open(DASHBOARD_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    logger.info("Saved → %s", DASHBOARD_JSON)
    return DASHBOARD_JSON


# ──────────────────────────────────────────────────────────────────────────────
# Tableau – Flat CSV bundle
# ──────────────────────────────────────────────────────────────────────────────

def save_tableau_csvs(
    corr_matrix,
    var_metrics,
    sharpe_metrics,
    log_returns,
    rolling_vol,
    terminal_values,
    weights,
    tickers=None,
    n_bins=100,
):
    """
    Export a bundle of Tableau-optimised flat CSV files into output/tableau/.

    Tableau conventions applied
    ---------------------------
    * Dates formatted as YYYY-MM-DD — Tableau auto-detects as Date dimension.
    * Boolean columns use True/False text for conditional colour encoding.
    * Field names use Title_Case_With_Underscores — displayed as pill labels.
    * Long / tidy format used for time-series charts with legend slicers.

    Files produced
    --------------
    tableau_correlation.csv       — Heatmap (Asset_X, Asset_Y, Correlation)
    tableau_kpi_summary.csv       — KPI cards (one row, all metrics as columns)
    tableau_asset_stats.csv       — Per-asset bar/table (one row per asset)
    tableau_returns_long.csv      — Multi-line returns (Date, Asset, Log_Return)
    tableau_rolling_vol_long.csv  — Rolling vol ribbon (Date, Asset, Ann_Vol)
    tableau_mc_distribution.csv   — MC histogram + VaR flag for colour split
    """
    if tickers is None:
        from alphapulse.config import TICKERS
        tickers = TICKERS

    _ensure_output_dir()
    tab_dir = os.path.join(OUTPUT_DIR, "tableau")
    os.makedirs(tab_dir, exist_ok=True)

    saved = {}
    n = len(tickers)

    # ── 1. Correlation heatmap ────────────────────────────────────────────
    corr_path = os.path.join(tab_dir, "tableau_correlation.csv")
    corr_rows = [
        {
            "Asset_X":     tickers[i],
            "Asset_Y":     tickers[j],
            "Correlation": round(_safe_float(corr_matrix[i, j]), 6),
        }
        for i in range(n) for j in range(n)
    ]
    pd.DataFrame(corr_rows).to_csv(corr_path, index=False)
    logger.info("Saved → %s", corr_path)
    saved["correlation"] = corr_path

    # ── 2. KPI summary (single row) ───────────────────────────────────────
    kpi_path = os.path.join(tab_dir, "tableau_kpi_summary.csv")
    kpi_row = {
        "Annualised_Return_Pct":     round(sharpe_metrics["annualised_return"] * 100, 4),
        "Annualised_Volatility_Pct": round(sharpe_metrics["annualised_volatility"] * 100, 4),
        "Sharpe_Ratio":              sharpe_metrics["sharpe_ratio"],
        "VaR_95_USD":                var_metrics["var_dollar"],
        "VaR_95_Pct":                var_metrics["var_pct"],
        "Initial_Portfolio_USD":     var_metrics["initial_value"],
        "VaR_Threshold_USD":         var_metrics["percentile_value"],
        "MC_Horizon_Days":           30,
        "Confidence_Level_Pct":      round(var_metrics["confidence_level"] * 100, 1),
        "Generated_At":              datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    pd.DataFrame([kpi_row]).to_csv(kpi_path, index=False)
    logger.info("Saved → %s", kpi_path)
    saved["kpi_summary"] = kpi_path

    # ── 3. Per-asset statistics ───────────────────────────────────────────
    asset_path = os.path.join(tab_dir, "tableau_asset_stats.csv")
    asset_rows = []
    for i, ticker in enumerate(tickers):
        col = log_returns[ticker].dropna()
        asset_rows.append({
            "Asset":                 ticker,
            "Weight_Pct":            round(float(weights[i]) * 100, 2),
            "Annual_Return_Pct":     round(float(col.mean()) * 252 * 100, 4),
            "Annual_Volatility_Pct": round(float(col.std(ddof=1)) * (252 ** 0.5) * 100, 4),
            "Daily_Volatility":      round(float(col.std(ddof=1)), 8),
            "Min_Daily_Return":      round(float(col.min()), 8),
            "Max_Daily_Return":      round(float(col.max()), 8),
            "Skewness":              round(float(col.skew()), 6),
            "Excess_Kurtosis":       round(float(col.kurt()), 6),
        })
    pd.DataFrame(asset_rows).to_csv(asset_path, index=False)
    logger.info("Saved → %s", asset_path)
    saved["asset_stats"] = asset_path

    # ── 4. Daily log returns – long format ────────────────────────────────
    returns_path = os.path.join(tab_dir, "tableau_returns_long.csv")
    lr = log_returns.copy()
    lr.index.name = "Date"
    returns_long = (
        lr.reset_index()
        .melt(id_vars="Date", var_name="Asset", value_name="Log_Return")
        .dropna(subset=["Log_Return"])
    )
    returns_long["Date"] = returns_long["Date"].astype(str).str[:10]
    returns_long["Log_Return"] = returns_long["Log_Return"].round(8)
    returns_long.to_csv(returns_path, index=False)
    logger.info("Saved → %s", returns_path)
    saved["returns_long"] = returns_path

    # ── 5. Rolling volatility – long format ──────────────────────────────
    vol_path = os.path.join(tab_dir, "tableau_rolling_vol_long.csv")
    rv = rolling_vol.copy()
    rv.index.name = "Date"
    vol_long = (
        rv.reset_index()
        .melt(id_vars="Date", var_name="Asset", value_name="Annualised_Volatility")
        .dropna(subset=["Annualised_Volatility"])
    )
    vol_long["Date"] = vol_long["Date"].astype(str).str[:10]
    vol_long["Annualised_Volatility"] = vol_long["Annualised_Volatility"].round(6)
    vol_long.to_csv(vol_path, index=False)
    logger.info("Saved → %s", vol_path)
    saved["rolling_vol_long"] = vol_path

    # ── 6. MC distribution with VaR threshold flag ───────────────────────
    mc_path = os.path.join(tab_dir, "tableau_mc_distribution.csv")
    counts, bin_edges = np.histogram(terminal_values, bins=n_bins)
    bin_midpoints = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.float64)
    var_threshold = float(var_metrics["percentile_value"])

    mc_df = pd.DataFrame({
        "Bin_Midpoint_USD":    bin_midpoints.round(2),
        "Frequency":           counts,
        "Probability":         (counts / counts.sum()).round(6),
        "Below_VaR_Threshold": bin_midpoints < var_threshold,
    })
    mc_df.to_csv(mc_path, index=False)
    logger.info("Saved → %s", mc_path)
    saved["mc_distribution"] = mc_path

    logger.info("Tableau CSV bundle complete: %d files in %s/", len(saved), tab_dir)
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# Tableau Public – Excel workbook  (single-file, multi-sheet)
# ──────────────────────────────────────────────────────────────────────────────

def save_tableau_excel(
    corr_matrix,
    var_metrics,
    sharpe_metrics,
    log_returns,
    rolling_vol,
    terminal_values,
    weights,
    tickers=None,
    n_bins: int = 100,
) -> str:
    """
    Export all Tableau-optimised datasets into a single multi-sheet Excel
    workbook: ``output/AlphaPulse_Tableau.xlsx``.

    Tableau Public accepts **Microsoft Excel** as a native data source.
    After opening the workbook you can switch between sheets in Tableau's
    Data Source tab without re-importing any files.

    Sheets
    ------
    KPI_Summary          — Portfolio-level KPI row (Sharpe, VaR, return, vol)
    Asset_Stats          — Per-asset statistics table
    Correlation          — Long-format heatmap (Asset_X, Asset_Y, Correlation)
    Returns_Long         — Daily log returns in tidy/long format
    Rolling_Vol_Long     — 30-day rolling volatility in tidy/long format
    MC_Distribution      — Monte Carlo histogram with VaR flag column
    """
    if tickers is None:
        from alphapulse.config import TICKERS
        tickers = TICKERS

    _ensure_output_dir()
    excel_path = os.path.join(OUTPUT_DIR, "AlphaPulse_Tableau.xlsx")
    n = len(tickers)

    # ── Sheet 1: KPI Summary ─────────────────────────────────────────────
    kpi_row = {
        "Annualised_Return_Pct":     round(sharpe_metrics["annualised_return"] * 100, 4),
        "Annualised_Volatility_Pct": round(sharpe_metrics["annualised_volatility"] * 100, 4),
        "Sharpe_Ratio":              round(sharpe_metrics["sharpe_ratio"], 6),
        "VaR_95_USD":                round(var_metrics["var_dollar"], 2),
        "VaR_95_Pct":                round(var_metrics["var_pct"], 4),
        "Initial_Portfolio_USD":     round(var_metrics["initial_value"], 2),
        "VaR_Threshold_USD":         round(var_metrics["percentile_value"], 2),
        "MC_Horizon_Days":           30,
        "Confidence_Level_Pct":      round(var_metrics["confidence_level"] * 100, 1),
        "Generated_At":              datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
    df_kpi = pd.DataFrame([kpi_row])

    # ── Sheet 2: Asset Stats ─────────────────────────────────────────────
    asset_rows = []
    for i, ticker in enumerate(tickers):
        col = log_returns[ticker].dropna()
        asset_rows.append({
            "Asset":                 ticker,
            "Weight_Pct":            round(float(weights[i]) * 100, 2),
            "Annual_Return_Pct":     round(float(col.mean()) * 252 * 100, 4),
            "Annual_Volatility_Pct": round(float(col.std(ddof=1)) * (252 ** 0.5) * 100, 4),
            "Daily_Volatility":      round(float(col.std(ddof=1)), 8),
            "Min_Daily_Return":      round(float(col.min()), 8),
            "Max_Daily_Return":      round(float(col.max()), 8),
            "Skewness":              round(float(col.skew()), 6),
            "Excess_Kurtosis":       round(float(col.kurt()), 6),
        })
    df_assets = pd.DataFrame(asset_rows)

    # ── Sheet 3: Correlation Heatmap ─────────────────────────────────────
    corr_rows = [
        {
            "Asset_X":     tickers[i],
            "Asset_Y":     tickers[j],
            "Correlation": round(_safe_float(corr_matrix[i, j]), 6),
        }
        for i in range(n) for j in range(n)
    ]
    df_corr = pd.DataFrame(corr_rows)

    # ── Sheet 4: Returns Long ────────────────────────────────────────────
    lr = log_returns.copy()
    lr.index.name = "Date"
    df_returns = (
        lr.reset_index()
        .melt(id_vars="Date", var_name="Asset", value_name="Log_Return")
        .dropna(subset=["Log_Return"])
    )
    df_returns["Date"] = pd.to_datetime(df_returns["Date"].astype(str).str[:10])
    df_returns["Log_Return"] = df_returns["Log_Return"].round(8)
    df_returns = df_returns.reset_index(drop=True)

    # ── Sheet 5: Rolling Volatility Long ────────────────────────────────
    rv = rolling_vol.copy()
    rv.index.name = "Date"
    df_vol = (
        rv.reset_index()
        .melt(id_vars="Date", var_name="Asset", value_name="Annualised_Volatility")
        .dropna(subset=["Annualised_Volatility"])
    )
    df_vol["Date"] = pd.to_datetime(df_vol["Date"].astype(str).str[:10])
    df_vol["Annualised_Volatility"] = df_vol["Annualised_Volatility"].round(6)
    df_vol = df_vol.reset_index(drop=True)

    # ── Sheet 6: Monte Carlo Distribution ───────────────────────────────
    counts, bin_edges = np.histogram(terminal_values, bins=n_bins)
    bin_midpoints = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.float64)
    var_threshold = float(var_metrics["percentile_value"])
    df_mc = pd.DataFrame({
        "Bin_Midpoint_USD":    bin_midpoints.round(2),
        "Frequency":           counts,
        "Probability":         (counts / counts.sum()).round(6),
        "Below_VaR_Threshold": bin_midpoints < var_threshold,
    })

    # ── Write workbook ───────────────────────────────────────────────────
    with pd.ExcelWriter(excel_path, engine="openpyxl", date_format="YYYY-MM-DD") as writer:
        df_kpi.to_excel(writer, sheet_name="KPI_Summary", index=False)
        df_assets.to_excel(writer, sheet_name="Asset_Stats", index=False)
        df_corr.to_excel(writer, sheet_name="Correlation", index=False)
        df_returns.to_excel(writer, sheet_name="Returns_Long", index=False)
        df_vol.to_excel(writer, sheet_name="Rolling_Vol_Long", index=False)
        df_mc.to_excel(writer, sheet_name="MC_Distribution", index=False)

    logger.info("Saved → %s  (6 sheets, Tableau Public ready)", excel_path)
    return excel_path
