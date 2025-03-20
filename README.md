# YieldCurveForecaster
A quantitative fixed-income analytics toolkit for visualizing, forecasting, and analyzing treasury yields. Implements Nelson-Siegel and heuristic models with exogenous factor integration. Features interactive 3D visualization, robust backtesting, and statistical metrics to support macroeconomic analysis.

## FRED API Setup

This project uses the Federal Reserve Economic Data (FRED) API to fetch financial data. To use the FRED API, you'll need:

- A FRED API key, which you can obtain for free from the St. Louis Fed: https://fred.stlouisfed.org/docs/api/api_key.html


## Project Setup

1. Clone the Repository:
   ```bash
   git clone https://github.com/yourusername/YieldCurveForecaster.git
   cd YieldCurveForecaster
```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
```
3. Set your API key as an environment variable:
   ```bash
   # On Linux/macOS
   export FRED_API_KEY='your_api_key_here'
   
   # On Windows (Command Prompt)
   set FRED_API_KEY=your_api_key_here
   
   # On Windows (PowerShell)
   $env:FRED_API_KEY='your_api_key_here'

```

4. Initialize the database (if not already present):
   ```bash
   python update_db.py
```

## Running the Dashboard

Launch the interactive dashboard:
   ```bash
   python dashboard.py
```
Navigate to http://127.0.0.1:8092/ in your browser to access the application.

## Project Structure

The project has the following components:

- `config.py`: Configuration parameters
- `data.py`: Data collection and processing functions
- `models.py`: Yield curve modeling functions
- `analysis.py`: Analysis and backtesting functions
- `plotting.py`: Visualization functions
- `utils.py`: Helper utilities
- `dashboard.py`: Interactive Dash application
- `update_db.py`: Database update script

## Models Implemented

1. **Heuristic**: Simple CPI-based regime-aware exponential decay forecasting model 
2. **Improved Heuristic**: Heursistic model with more sophisticated regime-awareness
3. **RLS (Random Level Shifts)**: Risk-premia adjusted forecasting model based on CBOE VIX
4. **Arbitrage-Free Nelson-Siegel**: Classic AFNS statistical model 