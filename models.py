import numpy as np
import pandas as pd
from scipy.optimize import minimize
import utils
import data
import config

def nelson_siegel(maturities, beta0, beta1, beta2, lambda_):
    """Implements the Nelson-Siegel yield curve math-model.
    
    Args:
        maturities: Array of maturities in years.
        beta0: Level parameter representing the long-term factor.
        beta1: Slope parameter representing the short-term factor.
        beta2: Curvature parameter for medium-term movements.
        lambda_: Decay parameter controlling factor loading shapes.
    
    Returns:
        Array of yields corresponding to input maturities.
    """
    term1 = (1 - np.exp(-lambda_ * maturities)) / (lambda_ * maturities)
    term2 = term1 - np.exp(-lambda_ * maturities)
    return beta0 + beta1 * term1 + beta2 * term2


def nelson_siegel_svensson(maturities, beta0, beta1, beta2, beta3, lambda_1, lambda_2):
    """Implements the Nelson-Siegel-Svensson yield curve math-model.
    
    An extension of the Nelson-Siegel model with an additional hump term
    for better fitting of complex yield curves.
    
    Args:
        maturities: Array of maturities in years.
        beta0: Level parameter representing the long-term factor.
        beta1: Slope parameter representing the short-term factor.
        beta2: Curvature parameter for first hump/trough.
        beta3: Curvature parameter for second hump/trough.
        lambda_1: First decay parameter.
        lambda_2: Second decay parameter.
    
    Returns:
        Array of yields corresponding to input maturities.
    """
    term1 = (1 - np.exp(-lambda_1 * maturities)) / (lambda_1 * maturities)
    term2 = term1 - np.exp(-lambda_1 * maturities)
    term3 = (1 - np.exp(-lambda_2 * maturities)) / (lambda_2 * maturities) - np.exp(-lambda_2 * maturities)
    return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

'''
[!] Note: The following loss functions are included for reference. I ended up using 
          the L-BFGS method to optimize the curves, but they would be implemented
          using a lambda function.
'''
def ns_loss(params, maturities, yields):
    """Calculates the sum of squared errors for Nelson-Siegel model fitting.
    
    Args:
        params: Tuple containing (beta0, beta1, beta2, lambda_).
        maturities: Array of maturities in years.
        yields: Observed yields to fit.
    
    Returns:
        Sum of squared differences between predicted and observed yields.
    """
    beta0, beta1, beta2, lambda_ = params
    predicted = nelson_siegel(maturities, beta0, beta1, beta2, lambda_)
    return np.sum((predicted - yields) ** 2)


def nss_loss(params, maturities, yields):
    """Calculates the sum of squared errors for Nelson-Siegel-Svensson model fitting.
    
    Args:
        params: Tuple containing (beta0, beta1, beta2, beta3, lambda_1, lambda_2).
        maturities: Array of maturities in years.
        yields: Observed yields to fit.
    
    Returns:
        Sum of squared differences between predicted and observed yields.
    """
    beta0, beta1, beta2, beta3, lambda_1, lambda_2 = params
    predicted = nelson_siegel_svensson(maturities, beta0, beta1, beta2, beta3, lambda_1, lambda_2)
    return np.sum((predicted - yields) ** 2)


def fit_nelson_siegel(df):
    """Fits the Nelson-Siegel model to historical yield curve data.
    
    For each date in the DataFrame, fits NS parameters using L-BFGS-B
    optimization to minimize the sum of squared errors.
    
    Args:
        df: DataFrame with dates as index and maturities as columns.
    
    Returns:
        DataFrame with fitted parameters (beta0, beta1, beta2, lambda_) for each date.
    """
    factors = []
    maturities = df.columns.to_numpy()
    for date, yields in df.iterrows():
        # Start with reasonable initial guesses
        initial_beta0 = max(yields.mean(), 0.01)
        initial_guess = [initial_beta0, -2, 2, 0.1]
        
        # Set bounds to ensure parameters remain in reasonable ranges
        # Note: beta0 must be positive as it represents the long-term level
        bounds = [(0.001, None), (None, None), (None, None), (0.001, 5)]
        
        try:
            # Directly minimize the sum of squared errors
            result = minimize(
                lambda p: np.sum((nelson_siegel(maturities, *p) - yields) ** 2),
                initial_guess, 
                bounds=bounds, 
                method='L-BFGS-B'
            )
            factors.append([date, *result.x])
        except:
            # Skip problematic dates
            continue
            
    factors_df = pd.DataFrame(factors, columns=["date", "beta0", "beta1", "beta2", "lambda_"])
    factors_df.set_index("date", inplace=True)
    return factors_df


def fit_nelson_siegel_svensson(df):
    """Fits the Nelson-Siegel-Svensson model to historical yield curve data.
    
    Idea:
    The NSS model offers greater flexibility for fitting complex curve shapes.
    
    Args:
        df: DataFrame with dates as index and maturities as columns.
    
    Returns:
        DataFrame with fitted parameters (beta0, beta1, beta2, beta3, lambda_1, lambda_2)
        for each date.
    """
    factors = []
    maturities = df.columns.to_numpy()
    for date, yields in df.iterrows():
        # Start with reasonable initial guesses
        initial_beta0 = max(yields.mean(), 0.01)
        initial_guess = [initial_beta0, -2, 2, 0, 0.5, 1.5]
        
        # Set bounds to ensure parameters remain in reasonable ranges
        bounds = [
            (0.001, None),   # beta0: positive (long-term level)
            (None, None),    # beta1: any value (slope)
            (None, None),    # beta2: any value (first curvature)
            (None, None),    # beta3: any value (second curvature)
            (0.001, 10),     # lambda_1: positive decay parameter
            (0.001, 10)      # lambda_2: positive decay parameter
        ]
        
        try:
            # Directly minimize the sum of squared errors
            result = minimize(
                lambda p: np.sum((nelson_siegel_svensson(maturities, *p) - yields) ** 2),
                initial_guess, 
                bounds=bounds, 
                method='L-BFGS-B'
            )
            factors.append([date, *result.x])
        except:
            # Skip problematic dates
            continue
            
    factors_df = pd.DataFrame(
        factors, 
        columns=["date", "beta0", "beta1", "beta2", "beta3", "lambda_1", "lambda_2"]
    )
    factors_df.set_index("date", inplace=True)
    return factors_df


def heuristic_forecast_yield_curve(db_file, start_date, end_date, maturities, forecast_steps=24, randomize=True, vix_data=None):
    """First heuristic model for forecasting yield curves.
    
    Idea:
    Implements a simple exponential decay model with noise adjustments and
    inflation-based decay bias. Assumes yields generally revert toward 
    a decay-adjusted level.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to forecast.
        forecast_steps: Number of months to forecast. Defaults to 24.
        randomize: Whether to add random noise to forecasts. Defaults to True.
        vix_data: Optional DataFrame with VIX data for volatility adjustment.
    
    Returns:
        DataFrame containing forecasted yields by date and maturity.
    """
    # Get historical yield curve
    hist_df = data.prepare_data(db_file, start_date, end_date)
    hist_df.columns = [round(float(col), 4) for col in hist_df.columns]
    last_hist = hist_df.iloc[-1]
    forecast_dates = pd.date_range(start=hist_df.index[-1] + pd.DateOffset(months=1),
                                   periods=forecast_steps, freq='M')
    
    # Set base decay parameters: these determine the base exponential decay
    # Higher decay means faster convergence to the long-term mean
    d_max = 0.05  # max decay for shortest maturity
    d_min = 0.01  # min decay for longest maturity
    m_min = min(maturities)
    m_max = max(maturities)
    
    # Get dynamic decay bias from inflation data - adjust decay based on inflation
    decay_bias = utils.compute_dynamic_decay_bias(db_file, start_date, end_date, vix_data=vix_data)
    
    forecast_data = []
    
    # For each maturity, compute an effective decay and forecast yields
    for m in maturities:
        # Scale decay rates linearly by maturity - shorter maturities decay faster
        if m_max != m_min:
            base_decay = d_max - (d_max - d_min) * ((m - m_min) / (m_max - m_min))
        else:
            base_decay = d_max
            
        # Adjust decay with dynamic bias based on inflation environment
        effective_decay = max(base_decay + decay_bias, 0.001)
        
        start_yield = last_hist.get(m, np.nan)
        if np.isnan(start_yield):
            continue
            
        for i, date in enumerate(forecast_dates):
            # Apply deterministic base decay - exponential convergence pattern
            forecast_y = start_yield * ((1 - effective_decay) ** (i + 1))
            
            # Only add noise if randomize is True
            if randomize:
                # Dynamic noise strength: more for shorter maturities
                noise_strength = 0.03 + 0.002 * (1 - (m / max(maturities)))
                noise = np.random.normal(0, noise_strength * start_yield)

                # Occasionally add larger shocks to simulate market events
                if np.random.rand() < 0.05:  # 5% chance of a large market shock
                    noise += np.random.normal(0, 0.02 * start_yield)
                
                forecast_y += noise

            forecast_data.append({
                'date': date,
                'maturity_numeric': round(m, 4),
                'yield': forecast_y
            })
            
    forecast_df = pd.DataFrame(forecast_data)
    return forecast_df


def improved_heuristic_forecast_yield_curve(db_file, start_date, end_date, maturities, forecast_steps=24, randomize=True, vix_data=None):
    """
    Idea:
    Extends the basic model with maturity-specific decay rates, mean reversion,
    regime detection, VIX-based volatility adjustments, and economic factor
    integration. Designed to better handle complex market environments.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to forecast.
        forecast_steps: Number of months to forecast. Defaults to 24.
        randomize: Whether to add random noise to forecasts. Defaults to True.
        vix_data: Optional DataFrame with VIX data for volatility adjustment.
    
    Returns:
        DataFrame containing forecasted yields by date and maturity.
    """
    # Get historical yield curve
    hist_df = data.prepare_data(db_file, start_date, end_date)
    hist_df.columns = [round(float(col), 4) for col in hist_df.columns]
    last_hist = hist_df.iloc[-1]
    
    # Generate forecast dates
    forecast_dates = pd.date_range(start=hist_df.index[-1] + pd.DateOffset(months=1),
                                   periods=forecast_steps, freq='M')
    
    # Calculate historical statistics for mean reversion targets
    hist_stats = utils.calculate_historical_means_and_volatility(db_file, start_date, end_date, maturities)
    
    # Detect current regime to adjust forecasting behavior
    regime_info = utils.detect_yield_regime(db_file, start_date, end_date, vix_data)
    
    # Calculate economic factor score to guide forecast direction
    econ_factors = utils.calculate_economic_factor_score(db_file, start_date, end_date, vix_data)
    
    # Set base decay parameters with maturity-specific adjustments
    d_max = 0.06  # Increased max decay for shortest maturity
    d_min = 0.01  # min decay for longest maturity
    m_min = min(maturities)
    m_max = max(maturities)
    
    forecast_data = []
    
    # For each maturity, compute forecasted yields
    for m in maturities:
        # Skip if no historical data for this maturity
        start_yield = last_hist.get(m, np.nan)
        if np.isnan(start_yield) or m not in hist_stats:
            continue
            
        # Get maturity-specific inflation bias
        decay_bias = utils.compute_dynamic_decay_bias(db_file, start_date, end_date, maturity=m, vix_data=vix_data)
        
        # Compute base decay rate with non-linear scaling for short-term rates
        # Non-linear power adjustment creates more realistic yield curve shapes
        if m <= 1:  # 1 year or less - faster decay initially
            base_decay = d_max - (d_max - d_min) * np.power((m - m_min) / (m_max - m_min), 0.7)
        else:
            base_decay = d_max - (d_max - d_min) * np.power((m - m_min) / (m_max - m_min), 0.9)
            
        # Adjust decay with dynamic bias - cap at small positive number
        effective_decay = max(base_decay + decay_bias, 0.001)
        
        # Get historical stats for this maturity
        stats = hist_stats[m]
        
        # Determine appropriate mean reversion target based on maturity and economic factors
        composite_score = econ_factors['composite_score']
        
        # Mean reversion targets vary by maturity - short-term rates more influenced by recent data
        # medium and long-term rates more influenced by long-term averages
        if m <= 0.25:  # 3 months
            # Short-term rates revert primarily to short-term average
            mean_target = 0.7 * stats['short_term_mean'] + 0.3 * stats['medium_term_mean']
            # Apply economic factor adjustment
            mean_target *= (1 + 0.05 * composite_score)  # Adjust by up to ±10%
            reversion_strength = 0.15  # Stronger reversion for short-term
        elif m <= 2:  # 2 years
            # Medium-short rates balance short and medium averages
            mean_target = 0.4 * stats['short_term_mean'] + 0.6 * stats['medium_term_mean']
            mean_target *= (1 + 0.04 * composite_score)
            reversion_strength = 0.10
        elif m <= 5:  # 5 years
            # Medium rates balance medium and long-term averages
            mean_target = 0.3 * stats['short_term_mean'] + 0.5 * stats['medium_term_mean'] + 0.2 * stats['long_term_mean']
            mean_target *= (1 + 0.03 * composite_score)
            reversion_strength = 0.08
        elif m <= 10:  # 10 years
            # Longer rates use more long-term average
            mean_target = 0.2 * stats['medium_term_mean'] + 0.8 * stats['long_term_mean']
            mean_target *= (1 + 0.02 * composite_score)
            reversion_strength = 0.05
        else:  # > 10 years
            # Very long rates use mostly long-term average
            mean_target = stats['long_term_mean']
            mean_target *= (1 + 0.01 * composite_score)
            reversion_strength = 0.03  # Weaker reversion for long-term
            
        # Adjust mean target based on recent trend, especially for shorter maturities
        if m <= 2:
            trend_adjustment = stats['recent_trend'] * (3 / max(1, m))  # Stronger for shorter maturities
            mean_target += min(max(trend_adjustment, -0.5), 0.5)  # Cap the adjustment
            
        # Additional regime-based adjustments - adjust behavior based on yield curve shape
        if regime_info['curve_regime'] != 'unknown':
            # In an inverted curve, adjust mean targets for different maturities
            if regime_info['is_inverted']:
                # During inversion, expect short-term rates to eventually fall
                if m <= 2:
                    mean_target *= 0.95  # Reduce target for short maturities
                
            # During policy tightening, short-term rates may rise more sharply
            if regime_info['policy_cycle'] == 'tightening' and m <= 2:
                mean_target *= 1.05  # Increase target
                
            # During policy loosening, expect rates to fall over time
            if regime_info['policy_cycle'] == 'loosening':
                mean_target *= 0.95
        
        # VIX regime adjustments - higher volatility during high VIX periods
        vix_volatility_factor = 1.0
        if vix_data is not None and not vix_data.empty and 'vix_regime' in regime_info:
            if regime_info['vix_regime'] == 'high_volatility':
                vix_volatility_factor = 1.5  # More volatility in high VIX environments
            elif regime_info['vix_regime'] == 'low_volatility':
                vix_volatility_factor = 0.7  # Less volatility in low VIX environments
        
        # Now forecast each point using a sequential approach (each forecast builds on previous)
        prev_yield = start_yield
        for i, date in enumerate(forecast_dates):
            # Start with exponential decay from previous point
            decayed_yield = prev_yield * (1 - effective_decay)
            
            # Add mean reversion component (stronger as we move forward in time)
            # Reversion strength increases with forecast horizon
            time_factor = min(1.0, (i + 1) / 12)  # Caps at 1.0 after 12 months
            adjusted_reversion = reversion_strength * time_factor
            
            # Calculate the mean reversion pull - key component that makes long-term
            # forecasts converge to economically sensible levels
            reversion_pull = adjusted_reversion * (mean_target - decayed_yield)
            
            # Combine decay and mean reversion
            forecast_y = decayed_yield + reversion_pull
            
            # Ensure yield doesn't go negative
            forecast_y = max(forecast_y, 0.01)
            
            # Add noise if requested
            if randomize:
                # Dynamic noise based on maturity, historical volatility, and VIX regime
                base_noise = stats['short_term_std'] * 0.5 * vix_volatility_factor
                
                # Scale noise by maturity (more for shorter maturities)
                maturity_factor = np.exp(-0.2 * m)  # Exponential decay with maturity
                
                # Scale noise by forecast horizon (more for longer horizons)
                horizon_factor = np.sqrt(min(1.0, (i + 1) / 12))
                
                # Combine factors for final noise level
                noise_level = base_noise * maturity_factor * horizon_factor
                
                # Generate noise with slight upward bias for positive values
                noise = np.random.normal(0.002, noise_level)
                
                # Add small chance of larger jumps/shocks to simulate market events
                if np.random.rand() < 0.05:  # 5% chance
                    shock_size = np.random.normal(0, noise_level * 3)
                    noise += shock_size
                
                forecast_y += noise
                
                # Ensure yield doesn't go negative after adding noise
                forecast_y = max(forecast_y, 0.01)
            
            # Add to forecast data
            forecast_data.append({
                'date': date,
                'maturity_numeric': round(m, 4),
                'yield': forecast_y
            })
            
            # Update previous yield for next iteration
            prev_yield = forecast_y
    
    forecast_df = pd.DataFrame(forecast_data)
    return forecast_df


def nss_forecast_yield_curve(db_file, start_date, end_date, maturities, forecast_steps=24, randomize=True, vix_data=None):
    """Nelson-Siegel-Svensson based forecasting model with macroeconomic adjustments.
    
    Idea:
    Uses enhanced decay rates based on VIX and inflation data, with inflation-based
    target yield floors. This model emphasizes the relationship between monetary policy,
    inflation expectations, and market volatility.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to forecast.
        forecast_steps: Number of months to forecast. Defaults to 24.
        randomize: Whether to add random noise to forecasts. Defaults to True.
        vix_data: Optional DataFrame with VIX data for volatility adjustment.
    
    Returns:
        DataFrame containing forecasted yields by date and maturity.
    """
    # Get historical yield curve
    hist_df = data.prepare_data(db_file, start_date, end_date)
    hist_df.columns = [round(float(col), 4) for col in hist_df.columns]
    
    # Last observed yields
    last_yields = hist_df.iloc[-1]
    
    # Generate forecast dates
    forecast_dates = pd.date_range(start=hist_df.index[-1] + pd.DateOffset(months=1),
                                  periods=forecast_steps, freq='M')
    
    # Get VIX level if available (recent average)
    current_vix = 20  # Default VIX level
    if vix_data is not None and not vix_data.empty:
        current_vix = vix_data.sort_values('date').tail(20)['vix'].mean()
    
    # Get recent inflation data
    inf_df = data.get_historical_inflation(db_file, start_date, end_date)
    current_inflation = 2.0  # Default inflation estimate
    if not inf_df.empty:
        current_inflation = inf_df["inflation_rate"].iloc[-1]
        
    # Calculate VIX and inflation factors for model adjustments
    vix_factor = current_vix / 20.0  # Normalized to 1.0 at VIX=20
    
    # Inflation impact factor - higher inflation should lead to higher rates
    # Cap to prevent extreme impacts
    inflation_factor = min(max(current_inflation / 2.0, 0.75), 1.5)
    
    # Set decay parameters with VIX and inflation adjustments
    # Higher decay rates = faster convergence to long-term means
    d_max = 0.15 * vix_factor  # Very high short-term decay rate, adjusts with VIX
    d_min = 0.04  # Higher long-term decay rate
    
    # Forecast data collection
    forecast_data = []
    
    # For each forecast date
    for i, date in enumerate(forecast_dates):
        # For each maturity
        for m in maturities:
            # Get the most recent yield
            current_yield = last_yields[m]
            
            # Calculate maturity-specific decay rate (much faster for shorter maturities)
            # This creates realistic curve shapes as forecasts evolve
            if m <= 0.25:  # Very short-term (3M)
                decay_rate = d_max * 1.2  # Extra high decay for shortest rates
            elif m <= 1:  # Short-term (up to 1Y)
                decay_rate = d_max
            elif m <= 2:  # Short-to-medium (1-2Y)
                decay_rate = d_max - (d_max - d_min) * ((m - 1) / 1)
            elif m <= 5:  # Intermediate (2-5Y)
                decay_rate = d_min * 1.5  # Higher than long-term
            else:  # Long-term (>5Y)
                decay_rate = d_min
            
            # Apply exponential decay
            time_in_years = (i + 1) / 12
            
            # Calculate inflation-adjusted target yield
            # This creates a floor based on inflation expectations
            # Key economic concept: real yields tend to normalize relative to inflation
            if m <= 1:
                # Short rates: stronger inflation impact
                inflation_floor = max(0.25, current_inflation * 0.5)
            elif m <= 5:
                # Medium rates: moderate inflation impact 
                inflation_floor = max(0.5, current_inflation * 0.7)
            else:
                # Long rates: full inflation impact with term premium
                inflation_floor = max(1.0, current_inflation * 0.9 + 0.5)
                
            # Calculate the forecast with floor
            raw_forecast = current_yield * np.exp(-decay_rate * time_in_years)
            
            # Apply inflation floor with smoothing that increases with time
            # This ensures long-term economic reasonableness
            floor_impact = min(1.0, time_in_years)  # Increases with forecast horizon
            forecast_y = raw_forecast * (1 - floor_impact) + max(raw_forecast, inflation_floor) * floor_impact
            
            # Add VIX and inflation-based volatility
            if randomize:
                # Combined volatility based on VIX and inflation uncertainty
                combined_factor = vix_factor * (0.8 + 0.2 * inflation_factor)
                
                # Base volatility with time scaling - uncertainty increases with horizon
                base_vol = 0.02 * combined_factor * np.sqrt(time_in_years)
                
                # Maturity-specific adjustment - shorter maturities more volatile
                maturity_adj = 1.0 if m <= 1 else (1.0 - 0.5 * min(1.0, (m - 1) / 9))
                
                # Final volatility calculation
                volatility = base_vol * maturity_adj * forecast_y * 0.3
                
                # Add noise
                noise = np.random.normal(0, volatility)
                forecast_y += noise
            
            # Ensure yield is positive
            forecast_y = max(forecast_y, 0.01)
            
            # Store the forecast
            forecast_data.append({
                'date': date,
                'maturity_numeric': round(m, 4),
                'yield': forecast_y
            })
    
    # Create and return the forecast DataFrame
    forecast_df = pd.DataFrame(forecast_data)
    return forecast_df


def forecast_yield_curve(forecast_model, db_file, start_date, end_date, maturities, forecast_steps=24, randomize=True, vix_data=None):
    """Generate forecasts for yield curves across multiple maturities using specified model.
    
    Args:
        forecast_model: Model to use ("Heuristic", "ImprovedHeuristic", "RLS").
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to forecast.
        forecast_steps: Number of months to forecast. Defaults to 24.
        randomize: Whether to add random noise to forecasts. Defaults to True.
        vix_data: Optional DataFrame with VIX data for volatility adjustment.
    
    Returns:
        DataFrame with forecasted yields.
        
    Raises:
        ValueError: If an unsupported forecast model is specified.
    """
    if forecast_model == "ImprovedHeuristic":
        return improved_heuristic_forecast_yield_curve(
            db_file, start_date, end_date, maturities, forecast_steps, randomize, vix_data
        )
    elif forecast_model == "RLS":
        return nss_forecast_yield_curve(
            db_file, start_date, end_date, maturities, forecast_steps, randomize, vix_data
        )
    elif forecast_model == "Heuristic":
        return heuristic_forecast_yield_curve(
            db_file, start_date, end_date, maturities, forecast_steps, randomize, vix_data
        )
    else:
        raise ValueError(f"Unsupported forecast model: {forecast_model}")


def forecast_individual_maturity(db_file, start_date, end_date, maturity, forecast_steps=24, randomize=True, model="Heuristic", vix_data=None):
    """Extract forecast for a specific maturity from the full forecast.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturity: Specific maturity to forecast in years.
        forecast_steps: Number of months to forecast. Defaults to 24.
        randomize: Whether to add random noise to forecasts. Defaults to True.
        model: Forecasting model to use. Defaults to "Heuristic".
        vix_data: Optional DataFrame with VIX data for volatility adjustment.
    
    Returns:
        DataFrame with forecasted yields for the specified maturity.
    """
    full_forecast_df = forecast_yield_curve(model, db_file, start_date, end_date, config.MATURITIES_LIST, forecast_steps, randomize, vix_data)
    maturity = round(float(maturity), 4)
    ts = full_forecast_df[full_forecast_df['maturity_numeric'] == maturity].copy()
    ts.sort_values('date', inplace=True)
    return ts