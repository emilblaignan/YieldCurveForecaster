# YieldCurveForecaster
A quantitative fixed-income analytics toolkit for visualizing, forecasting, and analyzing treasury yields. Implements Nelson-Siegel and novel heuristic models with exogenous factor integration. Features interactive 3D visualization, robust backtesting, and regime detection to support fixed-income investment decisions and macroeconomic analysis.

## FRED API Setup

This project uses the Federal Reserve Economic Data (FRED) API to fetch financial data. To use the FRED API, you'll need:

1. A FRED API key, which you can obtain for free from the St. Louis Fed: https://fred.stlouisfed.org/docs/api/api_key.html

2. Set your API key as an environment variable:
   ```bash
   # On Linux/macOS
   export FRED_API_KEY='your_api_key_here'
   
   # On Windows (Command Prompt)
   set FRED_API_KEY=your_api_key_here
   
   # On Windows (PowerShell)
   $env:FRED_API_KEY='your_api_key_here'

