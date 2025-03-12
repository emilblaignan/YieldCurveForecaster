import numpy as np
import pandas as pd
import data
import utils
import models
import config

def run_backtest(db_file, start_date, end_date, maturities, backtest_windows=None, forecast_horizons=None, model="Heuristic", vix_data=None):
    """Runs historical backtests on yield curve forecasting models.
    
    Idea:
    Evaluates model performance by generating forecasts at multiple historical points
    and comparing them to actual yield outcomes. Calculates standard error metrics
    as well as directional accuracy.
    
    Args:
        db_file: Path to SQLite database with yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to forecast.
        backtest_windows: Optional list of start dates for backtesting windows.
            If None, windows are automatically selected.
        forecast_horizons: Optional list of forecast horizons in months.
            Defaults to [1, 3, 6, 12].
        model: Forecasting model to use ("Heuristic", "ImprovedHeuristic", "RLS").
        vix_data: Optional DataFrame with VIX data.
        
    Returns:
        dict: Nested dictionary with backtest results organized by maturity 
            and forecast horizon, including error metrics and raw data.
    """
    if forecast_horizons is None:
        forecast_horizons = [1, 3, 6, 12]  # Standard forecast horizons in months
    
    # Get historical data
    hist_df = data.prepare_data(db_file, start_date, end_date)
    hist_dates = hist_df.index.sort_values()
    
    # Define backtest windows if not provided - this determines where we'll start each backtest
    if backtest_windows is None:
        # Skip the very end to ensure sufficient data for forecast evaluation
        longest_horizon = max(forecast_horizons)
        
        # Calculate sensible earliest and latest dates for backtesting
        earliest_idx = min(len(hist_dates)//4, 12)  # Skip initial period to have enough training data
        earliest_backtest = hist_dates[earliest_idx]  
        
        # Leave enough data at the end to evaluate the longest forecast horizon
        latest_idx = max(0, len(hist_dates) - longest_horizon - 3)
        latest_backtest = hist_dates[latest_idx]  
        
        # Create evenly spaced backtest windows (up to 12 points for efficiency)
        num_windows = min(12, (latest_idx - earliest_idx) // 2)
        if num_windows > 1:
            step = max(1, (latest_idx - earliest_idx) // num_windows)
            backtest_windows = [hist_dates[i] for i in range(earliest_idx, latest_idx, step)]
        else:
            # Fallback if we don't have enough data
            backtest_windows = [hist_dates[len(hist_dates) // 2]]
    
    # Get relevant VIX data for backtest periods if available
    backtest_vix_data = vix_data.copy() if vix_data is not None else None
    
    # Initialize results structure
    results = {}
    metrics = ['MSE', 'MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']
    
    # Run backtests for each maturity and horizon
    for maturity in maturities:
        maturity_results = {}
        for horizon in forecast_horizons:
            horizon_results = {metric: [] for metric in metrics}
            horizon_results['dates'] = []
            horizon_results['actuals'] = []
            horizon_results['forecasts'] = []
            
            for start_date in backtest_windows:
                # Get data up to the backtest start date (training data cutoff)
                backtest_cutoff = pd.to_datetime(start_date)
                training_data_end = backtest_cutoff.strftime("%Y-%m-%d")
                training_data_start = hist_dates[0].strftime("%Y-%m-%d")
                
                # Filter VIX data to only use what would have been available at forecast time
                backtest_vix = None
                if backtest_vix_data is not None:
                    backtest_vix = backtest_vix_data[backtest_vix_data['date'] <= backtest_cutoff].copy()
                
                # Generate forecast without randomization for consistent backtest results
                forecast = models.forecast_individual_maturity(
                    db_file, 
                    training_data_start, 
                    training_data_end, 
                    maturity, 
                    forecast_steps=max(forecast_horizons), 
                    randomize=False,  # Important: deterministic forecasts for backtesting
                    model=model,
                    vix_data=backtest_vix
                )
                
                # Find the actual yield outcome at the forecast horizon
                target_date = backtest_cutoff + pd.DateOffset(months=horizon)
                nearest_actual_date = hist_dates[hist_dates >= target_date].min()
                
                if pd.isna(nearest_actual_date) or maturity not in hist_df.columns:
                    continue
                
                actual_yield = hist_df.loc[nearest_actual_date, maturity]
                
                # Get forecast for the same horizon
                nearest_forecast_date = forecast['date'][forecast['date'] >= target_date].min()
                if pd.isna(nearest_forecast_date):
                    continue
                    
                forecast_yield = forecast[forecast['date'] == nearest_forecast_date]['yield'].values[0]
                
                # Calculate error metrics
                error = actual_yield - forecast_yield
                abs_error = abs(error)
                squared_error = error ** 2
                pct_error = abs(error / actual_yield) * 100 if actual_yield != 0 else np.nan
                
                # Calculate directional accuracy - important for trading strategies
                # Did the model correctly predict the direction of yield change?
                prev_actual_date = backtest_cutoff
                prev_actual_yield = hist_df.loc[prev_actual_date, maturity]
                
                actual_direction = 1 if (actual_yield > prev_actual_yield) else (-1 if actual_yield < prev_actual_yield else 0)
                forecast_direction = 1 if (forecast_yield > prev_actual_yield) else (-1 if forecast_yield < prev_actual_yield else 0)
                directional_match = 1 if actual_direction == forecast_direction else 0
                
                # Store all metrics for this backtest point
                horizon_results['MSE'].append(squared_error)
                horizon_results['MAE'].append(abs_error)
                horizon_results['RMSE'].append(np.sqrt(squared_error))
                horizon_results['MAPE'].append(pct_error)
                horizon_results['Directional_Accuracy'].append(directional_match)
                
                # Store data for detailed analysis
                horizon_results['dates'].append(nearest_actual_date)
                horizon_results['actuals'].append(actual_yield)
                horizon_results['forecasts'].append(forecast_yield)
            
            # Calculate aggregate metrics across all backtest windows
            for metric in ['MSE', 'MAE', 'RMSE', 'MAPE', 'Directional_Accuracy']:
                if horizon_results[metric]:
                    maturity_results[f'{horizon}m_{metric}'] = np.nanmean(horizon_results[metric])
                else:
                    maturity_results[f'{horizon}m_{metric}'] = np.nan
            
            # Calculate R2 (coefficient of determination) for overall fit quality
            if len(horizon_results['actuals']) > 1:
                actuals = np.array(horizon_results['actuals'])
                forecasts = np.array(horizon_results['forecasts'])
                # Standard R2 formula: 1 - (SSE/SST)
                # Floor at 0 to avoid negative values for very poor forecasts
                r2 = max(0, 1 - np.sum((actuals - forecasts) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
                maturity_results[f'{horizon}m_R2'] = r2
            else:
                maturity_results[f'{horizon}m_R2'] = np.nan
                
            # Store the raw backtest data for visualization and further analysis
            maturity_results[f'{horizon}m_data'] = {
                'dates': horizon_results['dates'],
                'actuals': horizon_results['actuals'],
                'forecasts': horizon_results['forecasts']
            }
                
        results[maturity] = maturity_results
    
    return results


def generate_descriptive_statistics(db_file, start_date, end_date, maturities):
    """Generates comprehensive statistical analysis of historical yield curves.
    
    Idea:
    Calculates distribution metrics, time-series characteristics, and other
    statistical properties for each maturity point on the yield curve.
    
    Args:
        db_file: Path to SQLite database with yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to analyze.
    
    Returns:
        pd.DataFrame: DataFrame with descriptive statistics for each maturity,
            with maturities as rows and statistics as columns.
    """
    hist_df = data.prepare_data(db_file, start_date, end_date)
    
    stats = {}
    for maturity in maturities:
        if maturity in hist_df.columns:
            series = hist_df[maturity].dropna()
            
            # Basic distribution statistics
            stats[maturity] = {
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min(),  # Total observed range
                'iqr': series.quantile(0.75) - series.quantile(0.25),  # Interquartile range
                '25%': series.quantile(0.25),
                '75%': series.quantile(0.75),
                
                # Time-series specific metrics
                'first_obs': series.iloc[0],  # First observed value
                'last_obs': series.iloc[-1],  # Most recent value
                'total_change': series.iloc[-1] - series.iloc[0],  # Absolute change over period
                'pct_change': ((series.iloc[-1] / series.iloc[0]) - 1) * 100 if series.iloc[0] != 0 else np.nan,  # Percentage change
                'volatility_annual': series.pct_change().std() * np.sqrt(12) * 100,  # Annualized volatility
                
                # Higher moments and sign distribution
                'skewness': series.skew(),  # Distribution asymmetry
                'kurtosis': series.kurtosis(),  # "Tailedness" of distribution
                'negative_months': (series < 0).sum(),  # Count of negative yield months
                'positive_months': (series > 0).sum(),  # Count of positive yield months
                'zero_months': (series == 0).sum(),  # Count of zero yield months
                
                # Serial correlation - useful for assessing predictability
                'autocorr_1': series.autocorr(1),  # 1-month lag autocorrelation
                
                # Recent trend (helpful for near-term forecasting)
                'recent_trend': series.iloc[-12:].diff().mean() if len(series) >= 12 else np.nan  # Average monthly change over past year
            }
    
    return pd.DataFrame(stats).transpose().round(4)


def analyze_turning_points(db_file, start_date, end_date, maturities, window=6):
    """Identifies local maxima and minima in yield curves over time.
    
    Idea:
    Locates yield peaks and troughs that may represent important
    cyclical turning points in interest rate markets.
    
    Args:
        db_file: Path to SQLite database with yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to analyze.
        window: Window size for detecting local extrema. Larger values
            identify more significant turning points. Defaults to 6.
        
    Returns:
        dict: Dictionary with turning points for each maturity, containing
            date, value, and type (peak or trough) information.
    """
    hist_df = data.prepare_data(db_file, start_date, end_date)
    
    results = {}
    for maturity in maturities:
        if maturity in hist_df.columns:
            series = hist_df[maturity].dropna()
            
            # Identify local maxima (peaks) using sliding window comparison
            # A point is a peak if it's the maximum value within window points on each side
            peaks = []
            for i in range(window, len(series) - window):
                if series.iloc[i] == max(series.iloc[i-window:i+window+1]):
                    peaks.append({
                        'date': series.index[i],
                        'value': series.iloc[i],
                        'type': 'peak'
                    })
            
            # Identify local minima (troughs) using sliding window comparison
            # A point is a trough if it's the minimum value within window points on each side  
            troughs = []
            for i in range(window, len(series) - window):
                if series.iloc[i] == min(series.iloc[i-window:i+window+1]):
                    troughs.append({
                        'date': series.index[i],
                        'value': series.iloc[i],
                        'type': 'trough'
                    })
            
            # Combine peaks and troughs and sort chronologically
            turning_points = sorted(peaks + troughs, key=lambda x: x['date'])
            
            results[maturity] = turning_points
    
    return results


def calculate_vix_yield_correlations(db_file, start_date, end_date, maturities, vix_data):
    """Calculates correlations between VIX and yield curve movements.
    
    Idea:
    Analyzes the relationship between market volatility (CBOE VIX) and
    Treasury yields across different volatility regimes and time scales.
    Helps identify flight-to-quality patterns and risk premium dynamics.
    
    Args:
        db_file: Path to SQLite database with yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to analyze.
        vix_data: DataFrame containing VIX data with 'date' and 'vix' columns.
        
    Returns:
        pd.DataFrame: DataFrame with correlations between VIX and yields
            across different regimes, with maturities as rows and correlation
            types as columns.
    """
    if vix_data is None or vix_data.empty:
        return pd.DataFrame()
    
    # Prepare yield data
    hist_df = data.prepare_data(db_file, start_date, end_date)
    
    # Align VIX data with yield dates (using nearest available VIX reading)
    vix_aligned = vix_data.set_index('date').reindex(hist_df.index, method='nearest')
    
    # Calculate correlations at different time scales and volatility regimes
    correlations = {}
    
    # For each maturity point on the yield curve
    for maturity in maturities:
        if maturity not in hist_df.columns:
            continue
            
        # Overall level correlation - relationship between yield levels and VIX levels
        # Negative values often indicate flight-to-quality dynamics
        level_corr = np.corrcoef(hist_df[maturity].values, vix_aligned['vix'].values)[0, 1]
        
        # Changes correlation - relationship between yield changes and VIX changes
        # More relevant for short-term market dynamics and risk-off episodes
        yield_changes = hist_df[maturity].diff().dropna()
        vix_changes = vix_aligned['vix'].diff().dropna()
        
        # Ensure indices align for valid correlation calculation
        common_idx = yield_changes.index.intersection(vix_changes.index)
        changes_corr = np.corrcoef(
            yield_changes.loc[common_idx].values, 
            vix_changes.loc[common_idx].values
        )[0, 1] if len(common_idx) > 2 else np.nan
        
        # High volatility regime correlation - behavior during market stress
        # VIX > 30 typically indicates significant market uncertainty
        high_vol_idx = vix_aligned[vix_aligned['vix'] > 30].index
        high_vol_corr = np.corrcoef(
            hist_df.loc[high_vol_idx, maturity].values, 
            vix_aligned.loc[high_vol_idx, 'vix'].values
        )[0, 1] if len(high_vol_idx) > 2 else np.nan
        
        # Low volatility regime correlation - behavior during calm markets
        # VIX < 15 typically indicates complacent markets
        low_vol_idx = vix_aligned[vix_aligned['vix'] < 15].index
        low_vol_corr = np.corrcoef(
            hist_df.loc[low_vol_idx, maturity].values, 
            vix_aligned.loc[low_vol_idx, 'vix'].values
        )[0, 1] if len(low_vol_idx) > 2 else np.nan
        
        correlations[maturity] = {
            'level_correlation': level_corr,
            'changes_correlation': changes_corr,
            'high_vol_correlation': high_vol_corr,
            'low_vol_correlation': low_vol_corr
        }
    
    return pd.DataFrame(correlations).transpose().round(4)