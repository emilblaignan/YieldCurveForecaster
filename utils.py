import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import data
import config

def compute_dynamic_decay_bias(db_file, start_date, end_date, maturity=None, target_inflation=2.0, vix_data=None):
    """Computes an inflation and volatility-adjusted decay bias for yield curve forecasting.
    
    Idea:
    Lower (more negative) bias values slow down the decay rate, which is appropriate
    in high inflation environments where yields tend to be more persistent.
    
    Args:
        db_file: Path to SQLite database with historical data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturity: Optional maturity in years to adjust sensitivity by tenor.
        target_inflation: Target inflation rate, defaults to 2.0%.
        vix_data: Optional DataFrame with VIX data for volatility adjustment.
        
    Returns:
        float: Decay bias value (more negative values reduce decay rate).
    """
    # Get inflation data
    inf_df = data.get_historical_inflation(db_file, start_date, end_date)
    
    if inf_df.empty:
        return -0.02  # default bias if no inflation data is available
    
    # Get most recent inflation and rolling average to determine environment
    recent_inflation = inf_df["inflation_rate"].iloc[-1]
    rolling_inflation = inf_df["inflation_rate"].tail(6).mean()  # 6-month average
    
    # Calculate inflation momentum - positive trend indicates accelerating inflation
    inflation_trend = inf_df["inflation_rate"].diff(3).tail(6).mean()  # Trend over last 6 months
    
    # Adjust bias based on VIX level - higher volatility means faster decay
    vix_factor = 1.0  # Default is no adjustment
    if vix_data is not None and not vix_data.empty:
        # Calculate average VIX over last 3 months
        recent_vix = vix_data.sort_values('date').tail(90)['vix'].mean()
        
        # Adjust decay based on volatility regime - higher VIX means more uncertainty
        # and typically faster mean reversion in yields
        if recent_vix > 30:  # High volatility regime
            vix_factor = 1.3  # Increase decay in high volatility
        elif recent_vix > 20:  # Medium volatility
            vix_factor = 1.1
        elif recent_vix < 15:  # Low volatility
            vix_factor = 0.9  # Decrease decay in low volatility
    
    # Base bias calculation with momentum factor - higher inflation = more negative bias
    # More negative bias means slower decay, reflecting sticky yields in high inflation
    if recent_inflation > 4:  # Very high inflation
        base_bias = -0.08 - (0.02 * inflation_trend if inflation_trend > 0 else 0)
    elif recent_inflation > 3:  # High inflation
        base_bias = -0.05 - (0.01 * inflation_trend if inflation_trend > 0 else 0)
    elif recent_inflation >= 1:  # Normal inflation
        base_bias = -0.03
    elif recent_inflation >= 0:  # Low inflation
        base_bias = -0.015
    else:  # Deflation
        base_bias = -0.005 + (0.005 * abs(inflation_trend) if inflation_trend < 0 else 0)
    
    # Adjust for maturity if provided - short-term yields are more sensitive to inflation
    # This reflects the economic reality that monetary policy (which affects short rates)
    # is more responsive to inflation changes than long-term yields
    if maturity is not None:
        if maturity <= 0.25:  # 3 months or less
            sensitivity = 1.5  # More sensitive
        elif maturity <= 2:   # 2 years or less
            sensitivity = 1.2
        elif maturity <= 5:   # 5 years or less
            sensitivity = 1.0  # Baseline sensitivity
        elif maturity <= 10:  # 10 years or less
            sensitivity = 0.7
        else:  # > 10 years
            sensitivity = 0.5  # Less sensitive
            
        # Apply sensitivity adjustment
        base_bias *= sensitivity
    
    # Apply VIX factor
    base_bias *= vix_factor
        
    return base_bias


def calculate_historical_means_and_volatility(db_file, start_date, end_date, maturities):
    """Calculates historical yield statistics across multiple time horizons.
    
    Idea:
    Computes means, standard deviations, trends and other statistics for each
    maturity over short, medium, and long-term horizons. These statistics
    serve as inputs for mean reversion targets in forecasting models.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturities: List of maturities in years to analyze.
    
    Returns:
        dict: Dictionary with maturity as key and dict of statistics as value.
    """
    hist_df = data.prepare_data(db_file, start_date, end_date)
    results = {}
    
    # Calculate lookback windows for different time horizons
    # Use min() to handle cases where historical data is limited
    long_lookback = min(60, len(hist_df))   # 5 years or max available
    medium_lookback = min(24, len(hist_df)) # 2 years or max available
    short_lookback = min(6, len(hist_df))   # 6 months or max available
    
    for maturity in maturities:
        if maturity in hist_df.columns:
            # Get different lookback windows for multi-horizon analysis
            series_long = hist_df[maturity].tail(long_lookback)
            series_medium = hist_df[maturity].tail(medium_lookback)
            series_short = hist_df[maturity].tail(short_lookback)
            
            # Compute normalized deviation - indicates how far current yield is from historical mean
            # Useful for assessing mean reversion potential
            norm_dev = (hist_df[maturity].iloc[-1] - series_long.mean()) / series_long.std() if series_long.std() > 0 else 0
            
            # Store comprehensive set of statistics for forecasting models
            results[maturity] = {
                'long_term_mean': series_long.mean(),
                'medium_term_mean': series_medium.mean(),
                'short_term_mean': series_short.mean(),
                'long_term_std': series_long.std(),
                'medium_term_std': series_medium.std(), 
                'short_term_std': series_short.std(),
                'last_value': hist_df[maturity].iloc[-1],
                'norm_deviation': norm_dev,  # Normalized deviation from mean
                'min': series_long.min(),
                'max': series_long.max(),
                # Calculate recent trend (annualized for easier interpretation)
                'recent_trend': series_short.diff().mean() * 12  # Annualized
            }
    
    return results


def detect_yield_regime(db_file, start_date, end_date, vix_data=None):
    """Detects the current yield curve regime to adjust forecast behavior.
    
    Idea:
    Analyzes yield curve shape, slope changes, and policy indicators to
    determine the current market regime. Understanding the regime helps
    forecast models adapt to different economic environments.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        vix_data: Optional DataFrame with VIX data.
        
    Returns:
        dict: Dictionary with regime information including curve state, policy cycle,
              inversion status, spread change rate, and volatility regime.
    """
    # Get historical yield curve spreads
    spreads_df = calculate_yield_curve_differentials(db_file, start_date, end_date)
    
    # Initialize with default values
    regime_info = {
        'curve_regime': 'unknown',
        'policy_cycle': 'unknown',
        'is_inverted': False,
        'spread_change': 0,
        'vix_regime': 'normal'
    }
    
    # Focus on 10Y-2Y spread - the most commonly used recession indicator
    if '10Y-2Y' in spreads_df.columns:
        # Get recent spread values
        recent_spreads = spreads_df['10Y-2Y'].tail(6)
        
        # Check for inversion (negative spread) - key recession predictor
        is_inverted = recent_spreads.mean() < 0
        regime_info['is_inverted'] = is_inverted
        
        # Calculate spread change rate to detect steepening or flattening
        spread_change = recent_spreads.diff().mean() * 12  # Annualized change
        regime_info['spread_change'] = spread_change
        
        # Determine curve regime based on inversion status and change direction
        if is_inverted:
            if spread_change > 0.05:
                # Inverted but getting less inverted - often happens near end of tightening cycle
                regime_info['curve_regime'] = "inverted_steepening"
            else:
                # Strongly inverted - typical of late-cycle tightening
                regime_info['curve_regime'] = "inverted_flat"
        else:
            if spread_change > 0.1:
                # Rapidly steepening - often occurs during easing cycles
                regime_info['curve_regime'] = "steepening"
            elif spread_change < -0.1:
                # Flattening - typical of tightening cycles
                regime_info['curve_regime'] = "flattening"
            else:
                # Normal positive slope with moderate changes
                regime_info['curve_regime'] = "normal"
                
        # Determine policy cycle based on inflation trend
        inf_df = data.get_historical_inflation(db_file, start_date, end_date)
        if not inf_df.empty:
            recent_inflation = inf_df["inflation_rate"].tail(6)
            # Calculate inflation momentum to infer policy direction
            inflation_trend = recent_inflation.diff().mean() * 12  # Annualized change
            
            if inflation_trend > 0.5:
                # Rising inflation suggests tightening monetary policy
                regime_info['policy_cycle'] = "tightening"
            elif inflation_trend < -0.5:
                # Falling inflation suggests easing monetary policy
                regime_info['policy_cycle'] = "loosening"
            else:
                # Stable inflation suggests neutral policy stance
                regime_info['policy_cycle'] = "neutral"
    
    # Classify volatility regime based on VIX level
    if vix_data is not None and not vix_data.empty:
        recent_vix = vix_data.sort_values('date').tail(20)['vix'].mean()
        if recent_vix > 30:
            regime_info['vix_regime'] = "high_volatility"
        elif recent_vix > 20:
            regime_info['vix_regime'] = "medium_volatility"
        elif recent_vix < 15:
            regime_info['vix_regime'] = "low_volatility"
            
    return regime_info


def calculate_economic_factor_score(db_file, start_date, end_date, vix_data=None):
    """Calculates a composite economic factor score for yield forecast adjustments.
    
    Idea:
    Combines inflation, yield curve shape, and market volatility signals into
    a single score that guides yield forecasts. Higher scores generally suggest
    upward pressure on yields.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        vix_data: Optional DataFrame with VIX data.
    
    Returns:
        dict: Dictionary with individual factor scores and composite score.
    """
    # Initialize with neutral scores for all factors
    factors = {
        'inflation_score': 0,
        'curve_score': 0, 
        'vix_score': 0,
        'composite_score': 0
    }
    
    # Calculate inflation score - higher inflation generally leads to higher yields
    inf_df = data.get_historical_inflation(db_file, start_date, end_date)
    if not inf_df.empty:
        recent_inflation = inf_df["inflation_rate"].iloc[-1]
        # Score from -2 (very low inflation) to +2 (very high inflation)
        if recent_inflation > 5:  # Very high inflation
            factors['inflation_score'] = 2
        elif recent_inflation > 3:  # Above-target inflation
            factors['inflation_score'] = 1
        elif recent_inflation < 1:  # Low inflation
            factors['inflation_score'] = -1
        elif recent_inflation < 0:  # Deflation
            factors['inflation_score'] = -2
    
    # Calculate yield curve score - inverted curves typically precede lower future yields
    regime_info = detect_yield_regime(db_file, start_date, end_date)
    # Score from -2 (deeply inverted) to +2 (steeply positive)
    if regime_info['curve_regime'] == 'inverted_flat':
        factors['curve_score'] = -2  # Strongly inverted - recession risk
    elif regime_info['curve_regime'] == 'inverted_steepening':
        factors['curve_score'] = -1  # Moving away from inversion - easing expected
    elif regime_info['curve_regime'] == 'steepening':
        factors['curve_score'] = 2   # Strongly steepening - growth/inflation expected
    elif regime_info['curve_regime'] == 'flattening':
        factors['curve_score'] = -0.5  # Flattening - tightening cycle
    
    # Calculate VIX score - higher volatility can lead to flight-to-quality
    # which typically lowers Treasury yields (especially long-term)
    if vix_data is not None and not vix_data.empty:
        recent_vix = vix_data.sort_values('date').tail(20)['vix'].mean()
        # Score from -2 (very low volatility) to +2 (very high volatility)
        if recent_vix > 35:  # Crisis-level volatility
            factors['vix_score'] = 2
        elif recent_vix > 25:  # Elevated volatility
            factors['vix_score'] = 1
        elif recent_vix < 15:  # Low volatility
            factors['vix_score'] = -1
        elif recent_vix < 12:  # Very low volatility
            factors['vix_score'] = -2
    
    # Calculate weighted composite score - inflation and curve shape have
    # larger impacts on yields than volatility
    factors['composite_score'] = (
        factors['inflation_score'] * 0.4 + 
        factors['curve_score'] * 0.4 + 
        factors['vix_score'] * 0.2
    )
    
    return factors


def calculate_yield_curve_differentials(db_file, start_date, end_date, maturity_pairs=None):
    """Calculates common yield curve spreads and identifies inversions.
    
    Idea:
    Computes spreads between key maturity points on the yield curve,
    which are important indicators of economic expectations and cycles.
    
    Args:
        db_file: Path to SQLite database with historical yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturity_pairs: Optional list of tuples with (long_term, short_term) 
                        maturities to calculate spreads.
        
    Returns:
        DataFrame: DataFrame with yield curve differentials and inversion flags.
    """
    # Default maturity pairs to analyze - each has specific economic significance
    if maturity_pairs is None:
        maturity_pairs = [
            (10, 2),    # 10Y-2Y - classic recession indicator
            (10, 0.25), # 10Y-3M - alternative recession indicator (Fed preferred)
            (30, 5),    # 30Y-5Y - long term expectations vs. medium-term
            (5, 2),     # 5Y-2Y - medium term expectations
            (2, 0.25)   # 2Y-3M - near term monetary policy expectations
        ]
    
    # Get historical yield data
    hist_df = data.prepare_data(db_file, start_date, end_date)
    
    # Create DataFrame to store spreads
    spreads_df = pd.DataFrame(index=hist_df.index)
    
    for long_term, short_term in maturity_pairs:
        # Check if both maturities are available in the data
        if long_term in hist_df.columns and short_term in hist_df.columns:
            # Calculate spread and create appropriate label
            spread_name = f"{long_term}Y-{short_term}Y" if short_term >= 1 else f"{long_term}Y-{int(short_term*12)}M"
            spreads_df[spread_name] = hist_df[long_term] - hist_df[short_term]
            
            # Flag inversions (when spread is negative)
            # Inverted curves are historically strong recession predictors
            spreads_df[f"{spread_name}_inverted"] = spreads_df[spread_name] < 0
    
    return spreads_df