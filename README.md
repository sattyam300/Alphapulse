Here is a detailed breakdown of the **AlphaPulse** project:

### 1. Project Overview (In Points)
* **What it is:** AlphaPulse is a production-grade Python quantitative engine designed for cross-asset portfolio risk and volatility analysis.
* **Core Technology:** Built purely on Python and heavily optimized using **NumPy** for vectorized, loop-free mathematical calculations. 
* **Key Metrics Calculated:** It computes critical financial indicators including Daily Log Returns, Annualized Volatility, Sharpe Ratio, and Value at Risk (VaR).
* **Data Source:** Automatically fetches up-to-date historical market data (default: last 3 years) using Yahoo Finance (`yfinance`).
* **Output:** Generates clean, structured data files (CSV and JSON) that are specifically tailored to be imported into visualization tools like Tableau or Power BI.

### 2. How the Project Works (The Pipeline)
The project runs in a streamlined, automated pipeline divided into three main stages:

* **Stage 1: Data Acquisition (`data_acquisition.py`)**
  * Downloads historical "Adjusted Close" prices for a predefined basket of assets (e.g., AAPL, TSLA, BTC, GLD).
  * Applies **Forward Fill** to handle missing data on non-trading days (like weekends or holidays), ensuring time-series data is perfectly aligned without introducing look-ahead bias.
* **Stage 2: Quantitative Engine (`quant_engine.py`)**
  * **Log Returns:** Converts raw prices into daily log returns, which are statistically robust.
  * **Covariance Matrix:** Calculates how different assets move in relation to one another using highly optimized matrix multiplication. 
  * **Monte Carlo Simulation:** Uses Cholesky decomposition to run 10,000 simulated future paths for the portfolio over a 30-day horizon to model potential outcomes based on historical volatility.
  * **Risk Metrics:** Extracts the 95% Value at Risk (VaR) and the Sharpe ratio from these calculations.
* **Stage 3: Output Formatting (`output_formatter.py`)**
  * Takes the complex mathematical arrays and serializes them into dashboard-ready formats like `risk_metrics.json`, `rolling_volatility.csv`, and a master `dashboard_payload.json`.

### 3. What is the Use of the Project?
* **Risk Management:** It allows investors or portfolio managers to quantify their downside risk. For example, the 95% Value at Risk (VaR) tells a user: *"With 95% confidence, this portfolio will not lose more than $X over the next 30 days."*
* **Performance Evaluation:** By calculating the **Sharpe Ratio**, it helps users understand if they are taking on too much risk for the returns they are getting (risk-adjusted performance).
* **Asset Correlation:** Helps users see if their portfolio is truly diversified. If all assets are highly correlated, a market downturn will hit the entire portfolio simultaneously.
* **Dashboard Automation:** Instead of manually crunching numbers in Excel, this project serves as a backend engine that can automatically feed live, daily risk assessments into visual dashboards (Power BI / Tableau) for executive reporting.
