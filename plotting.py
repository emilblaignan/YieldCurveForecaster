import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import gc
import data
import models
import utils
import config

def generate_surface_plot(forecast_model, selected_maturities, db_file, start_date, end_date, vix_data=None):
    """Creates a 3D surface visualization of historical and forecasted yield curves.
    
    Purpose:
    Generates an interactive 3D surface plot showing the evolution of the yield curve
    over time, including both historical data and forward-looking model forecasts.
    Selected maturities can be highlighted with individual time series lines.
    
    Args:
        forecast_model: Model name to use for forecasting ("Heuristic", 
            "ImprovedHeuristic", or "RLS").
        selected_maturities: List of maturities to highlight with individual lines.
        db_file: Path to SQLite database with yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        vix_data: Optional DataFrame with VIX data for forecast enhancement.
        
    Returns:
        go.Figure: Plotly figure object containing the 3D yield curve visualization.
        
    Raises:
        Exception: If an error occurs during plot generation.
    """
    try:
        # Process historical data with memory efficiency in mind
        historical_df = data.prepare_data(db_file, start_date, end_date)
        historical_df.index = pd.to_datetime(historical_df.index)
        
        # For large datasets, subsample to improve performance while preserving trends
        # This is important for interactive visualizations to remain responsive
        if len(historical_df) > 200:
            historical_df = historical_df.iloc[::2]  # Take every other row
            
        # Prepare maturity data for consistent plotting
        available_maturities = [round(float(col), 4) for col in historical_df.columns]
        sorted_maturities = sorted(available_maturities, reverse=True)
        historical_df = historical_df.reindex(columns=config.MATURITIES_LIST)[sorted_maturities]
        
        # Convert dates to numeric values for 3D plotting
        base_date = historical_df.index.min()
        hist_dates_numeric = (historical_df.index - base_date).days
        
        # Create meshgrid for 3D surface - positions represent maturity indices
        positions = np.arange(len(sorted_maturities))
        X_hist, Y_hist = np.meshgrid(positions, hist_dates_numeric)
        Z_hist = historical_df.fillna(0).values
        
        # Adapt forecast length based on visualization complexity
        # Reduces memory usage and computation time for complex displays
        forecast_steps = 24  # Standard value: 2-year forecast
        if len(selected_maturities) > 5:
            # Shorter forecast horizon when displaying many maturity lines
            forecast_steps = 12
            
        # Generate forecast data using the selected model
        forecast_df = models.forecast_yield_curve(
            forecast_model, db_file, start_date, end_date, 
            sorted_maturities, forecast_steps=forecast_steps, vix_data=vix_data
        )
        
        # Prepare forecast data for 3D visualization
        forecast_df["maturity_numeric"] = forecast_df["maturity_numeric"].astype(float)
        forecast_pivot = forecast_df.pivot(index="date", columns="maturity_numeric", values="yield")
        forecast_pivot = forecast_pivot.reindex(columns=sorted_maturities).ffill().bfill().fillna(0)
        
        # Convert forecast dates to numeric values for plotting
        forecast_dates_numeric = (forecast_pivot.index - base_date).days.values
        X_fore, Y_fore = np.meshgrid(positions, forecast_dates_numeric)
        Z_fore = forecast_pivot.values

        # Adapt surface opacity based on selection mode:
        # - With selected maturities: Use transparent surfaces to highlight lines
        # - Without selections: Use opaque surfaces to show overall shape
        surface_opacity = 0.05 if selected_maturities else 0.6

        # Create the base figure
        fig = go.Figure()
        
        # Add historical yield curve surface (blue gradient)
        fig.add_trace(go.Surface(
            x=X_hist, y=Y_hist, z=Z_hist,
            colorscale="Blues",
            opacity=surface_opacity,
            name="Historical Yield Curve",
            showscale=False
        ))
        
        # Add forecast yield curve surface (orange-cyan-blue gradient)
        # Custom colors highlight the transition from historical to forecast data
        custom_scale = [[0, 'darkorange'], [0.5, 'cyan'], [1, 'darkblue']]
        fig.add_trace(go.Surface(
            x=X_fore, y=Y_fore, z=Z_fore,
            colorscale=custom_scale,
            opacity=surface_opacity,
            name="Forecast Yield Curve",
            showscale=False
        ))
        
        # Add individual time series lines for selected maturities
        # These stand out against the transparent surfaces when specific tenors are selected
        if selected_maturities:
            for m in selected_maturities:
                m_val = round(float(m), 4)
                if m_val not in sorted_maturities:
                    continue
                
                # Extract historical data for this maturity
                hist_ts = historical_df[m_val].reset_index()
                hist_ts.rename(columns={m_val: "yield"}, inplace=True)
                
                # Get forecast for this specific maturity
                forecast_ts = models.forecast_individual_maturity(
                    db_file, start_date, end_date, m_val, 
                    model=forecast_model, vix_data=vix_data
                )
                
                # Combine historical and forecast for continuous visualization
                combined_ts = pd.concat([hist_ts, forecast_ts], ignore_index=True)
                combined_ts.sort_values("date", inplace=True)
                combined_ts['date_str'] = combined_ts['date'].dt.strftime("%m-%Y")
                combined_ts['date_numeric'] = (combined_ts['date'] - base_date).dt.days
                
                # Add the maturity-specific line to the 3D plot
                pos = sorted_maturities.index(m_val)
                fig.add_trace(go.Scatter3d(
                    x=[pos] * len(combined_ts),  # Fixed x position for this maturity
                    y=combined_ts['date_numeric'],  # Date position
                    z=combined_ts['yield'],  # Yield values
                    mode='lines',
                    line=dict(color='black', width=1.1),
                    name=f"{int(m_val)}Y" if m_val >= 1 else f"{int(round(m_val*12))}M",
                    # Enhanced hover information for financial analysis
                    hovertemplate="<b>Maturity:</b> " + f"{m_val} years" +
                                "<br><b>Date:</b> %{customdata[0]}" +
                                "<br><b>Yield:</b> %{z:.2f}%",
                    customdata=combined_ts[['date_str']].values
                ))
        
        # Configure time axis with year labels
        end_date_val = forecast_pivot.index.max() if not forecast_pivot.empty else historical_df.index.max()
        year_ticks = pd.date_range(start=base_date, end=end_date_val, freq='YS')
        y_tick_vals = [(date - base_date).days for date in year_ticks]
        y_tick_text = [date.strftime("%Y") for date in year_ticks]
            
        # Set layout with optimized aspect ratio and clear axis labels
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="Maturity",
                    tickvals=list(range(len(sorted_maturities))),
                    ticktext=[f"{int(m)}Y" if m >= 1 else f"{int(round(m*12))}M" for m in sorted_maturities]
                ),
                yaxis=dict(
                    title="Time",
                    tickvals=y_tick_vals,
                    ticktext=y_tick_text
                ),
                zaxis=dict(title="Yield (%)"),
                # Aspect ratio optimized for yield curve visualization
                aspectratio=dict(x=1, y=2, z=0.7)
            ),
            title_text=f"Historical & Forecast Yield Curves ({forecast_model} Model)",
            margin=dict(l=0, r=0, b=10, t=30),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Clean up large intermediate objects to free memory
        del X_hist, Y_hist, Z_hist, X_fore, Y_fore, Z_fore
        gc.collect()
        
        return fig
    except Exception as e:
        # Log the error for debugging
        print(f"Error in generate_surface_plot: {str(e)}")
        raise


def plot_yield_spread_trends(db_file, start_date, end_date):
    """Creates a visualization of key yield curve spreads over time.
    
    Purpose:
    Plots the evolution of important yield curve spreads, which are critical indicators
    for economic cycle analysis, recession forecasting, and monetary policy expectations.
    
    Args:
        db_file: Path to SQLite database with yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        
    Returns:
        go.Figure: Plotly figure object with spread time series and zero line reference.
    """
    # Calculate yield curve spreads using utility function
    spreads_df = utils.calculate_yield_curve_differentials(db_file, start_date, end_date)
    
    # Handle case with no available spread data
    if spreads_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No spread data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create base figure
    fig = go.Figure()
    
    # Add each spread series (excluding inversion flags)
    for col in [col for col in spreads_df.columns if not col.endswith('_inverted')]:
        fig.add_trace(
            go.Scatter(
                x=spreads_df.index,
                y=spreads_df[col],
                mode='lines',
                name=col
            )
        )
    
    # Add zero line - crucial reference for identifying yield curve inversions
    # Inversions (negative spreads) are historically strong recession indicators
    fig.add_shape(
        type="line",
        x0=spreads_df.index.min(),
        y0=0,
        x1=spreads_df.index.max(),
        y1=0,
        line=dict(color="red", width=1, dash="dash")
    )
    
    # Configure layout with appropriate titles and dimensions
    fig.update_layout(
        title="Yield Curve Spreads Over Time",
        xaxis_title="Date",
        yaxis_title="Spread (%)",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_turning_points(db_file, start_date, end_date, maturity):
    """Visualizes local maxima and minima (turning points) in yield curve history.
    
    Purpose:
    Identifies and plots significant turning points in the yield curve for a specific
    maturity, helping analysts identify cyclical patterns and potential future
    reversal points.
    
    Args:
        db_file: Path to SQLite database with yield data.
        start_date: Start date for historical data in 'YYYY-MM-DD' format.
        end_date: End date for historical data in 'YYYY-MM-DD' format.
        maturity: Specific maturity point in years to analyze.
        
    Returns:
        go.Figure: Plotly figure showing yield curve with highlighted peaks and troughs.
    """
    # Identify turning points for the selected maturity
    turning_points = utils.analyze_turning_points(db_file, start_date, end_date, [maturity])
    
    # Handle case with no turning points identified
    if not turning_points or maturity not in turning_points:
        fig = go.Figure()
        fig.add_annotation(text="No turning points found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Get historical yield data for the selected maturity
    hist_df = data.prepare_data(db_file, start_date, end_date)
    if maturity not in hist_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Historical data not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create base figure
    fig = go.Figure()
    
    # Plot the complete yield time series for context
    fig.add_trace(
        go.Scatter(
            x=hist_df.index,
            y=hist_df[maturity],
            mode='lines',
            name=f"{int(maturity)}Y" if maturity >= 1 else f"{int(round(maturity*12))}M"
        )
    )
    
    # Separate the turning points into peaks and troughs for distinct visualization
    peaks = [tp for tp in turning_points[maturity] if tp['type'] == 'peak']
    troughs = [tp for tp in turning_points[maturity] if tp['type'] == 'trough']
    
    # Add peaks (local maxima) with upward triangle markers
    if peaks:
        peak_dates = [p['date'] for p in peaks]
        peak_values = [p['value'] for p in peaks]
        
        fig.add_trace(
            go.Scatter(
                x=peak_dates,
                y=peak_values,
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-up'),
                name='Peaks'
            )
        )
    
    # Add troughs (local minima) with downward triangle markers
    if troughs:
        trough_dates = [t['date'] for t in troughs]
        trough_values = [t['value'] for t in troughs]
        
        fig.add_trace(
            go.Scatter(
                x=trough_dates,
                y=trough_values,
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-down'),
                name='Troughs'
            )
        )
    
    # Format maturity label for display
    maturity_label = f"{int(maturity)}Y" if maturity >= 1 else f"{int(round(maturity*12))}M"
    
    # Configure layout with appropriate titles and dimensions
    fig.update_layout(
        title=f"Turning Points for {maturity_label} Yield",
        xaxis_title="Date",
        yaxis_title="Yield (%)",
        height=600
    )
    
    return fig