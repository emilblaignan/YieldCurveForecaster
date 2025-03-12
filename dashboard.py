import pandas as pd
import numpy as np
import gc
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import data
import models
import analysis
import plotting
import utils
import config

# Load VIX data for volatility-based adjustments to forecasts
# This is optional - models will adjust when available
vix_data = data.query_vix_data(config.DB_FILE, config.DEFAULT_START_DATE, config.DEFAULT_END_DATE)

def create_app():
    """Creates and configures the Dash application for yield curve analysis.
    
    Purpose:
    Constructs a multi-tab dashboard with visualization, backtesting, and analysis
    components. Establishes all necessary callbacks for interactivity.
    
    Returns:
        Dash: Configured Dash application ready to be run by a server.
    """
    # Initialize Dash app with Bootstrap styling for professional appearance
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Define the application layout with multiple tabs
    app.layout = html.Div([
        # Navigation header
        dbc.NavbarSimple(
            brand="Yield Curve Analysis & Forecasting Dashboard",
            brand_href="#",
            color="primary",
            dark=True,
        ),
        
        # State storage for maintaining app state across callback executions
        dcc.Store(id='current-model-store', data='Heuristic'),  # Default model
        dcc.Store(id='selected-maturities-store', data=[]),     # Selected tenors
        dcc.Store(id='backtest-state-store', data={}),          # Backtest results
        dcc.Store(id='viz-needs-refresh', data=False),          # Visualization refresh flag
        
        # Main tab structure - separates visualization, backtesting, and analysis
        dbc.Tabs([
            # Tab 1: 3D Yield Curve Visualization -------------------------
            dbc.Tab(label="Yield Curve Visualization", tab_id="tab-visualization", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Model Configuration", className="mt-3"),
                        dbc.Row([
                            # Forecast model selection - determines the algorithm used
                            dbc.Col([
                                html.Label("Select Forecast Model:"),
                                dcc.Dropdown(
                                    id="forecast-model-dropdown",
                                    options=[
                                        {"label": "Heuristic", "value": "Heuristic"},
                                        {"label": "Improved Heuristic", "value": "ImprovedHeuristic"},
                                        {"label": "RLS", "value": "RLS"}
                                    ],
                                    value="Heuristic",
                                    clearable=False
                                )
                            ], width=6),
                            # Maturity selection - highlights specific tenors in the 3D view
                            dbc.Col([
                                html.Label("Select Maturities to Highlight:"),
                                dcc.Dropdown(
                                    id="maturities-dropdown",
                                    options=[{"label": f"{int(m)}Y" if m >= 1 else f"{int(round(m*12))}M", "value": m}
                                             for m in sorted(config.MATURITIES_LIST, reverse=True)],
                                    multi=True,
                                    value=[]
                                )
                            ], width=6)
                        ]),
                        # Main 3D visualization container
                        html.Div(id="visualization-container", className="mt-3", children=[
                            dcc.Graph(id="yield-curve-graph", style={"height": "70vh"})
                        ])
                    ])
                ])
            ], id="tab-visualization"),
            
            # Tab 2: Backtesting & Model Evaluation -------------------------
            dbc.Tab(label="Backtesting & Evaluation", tab_id="tab-backtest", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Backtest Configuration", className="mt-3"),
                        dbc.Row([
                            # Maturity selection for backtest - which tenors to evaluate
                            dbc.Col([
                                html.Label("Select Maturities:"),
                                dcc.Dropdown(
                                    id="backtest-maturities-dropdown",
                                    options=[
                                        {"label": "Select All", "value": "all"},
                                        *[{"label": f"{int(m)}Y" if m >= 1 else f"{int(round(m*12))}M", "value": m}
                                         for m in sorted(config.MATURITIES_LIST, reverse=True)]
                                    ],
                                    multi=True,
                                    value=[10, 2]  # Default to 10Y and 2Y (common benchmark tenors)
                                )
                            ], width=6),
                            # Forecast horizon selection - how far ahead to test
                            dbc.Col([
                                html.Label("Select Forecast Horizons:"),
                                dcc.Dropdown(
                                    id="forecast-horizons-dropdown",
                                    options=[
                                        {"label": "1 Month", "value": 1},
                                        {"label": "3 Months", "value": 3},
                                        {"label": "6 Months", "value": 6},
                                        {"label": "12 Months", "value": 12}
                                    ],
                                    multi=True,
                                    value=[1, 6, 12]  # Default to common trading horizons
                                )
                            ], width=6)
                        ]),
                        dbc.Row([
                            # Model selection for backtest
                            dbc.Col([
                                html.Label("Select Forecast Model:"),
                                dcc.Dropdown(
                                    id="backtest-model-dropdown",
                                    options=[
                                        {"label": "Heuristic", "value": "Heuristic"},
                                        {"label": "Improved Heuristic", "value": "ImprovedHeuristic"},
                                        {"label": "RLS", "value": "RLS"}
                                    ],
                                    value="Heuristic",
                                    clearable=False
                                )
                            ], width=6)
                        ], className="mt-3"),
                        # Run button and results container with loading spinner
                        html.Div(className="d-grid gap-2 d-md-flex justify-content-md-end mt-3", children=[
                            dbc.Button("Run Backtest", id="run-backtest-btn", color="primary")
                        ]),
                        dbc.Spinner(
                            html.Div(id="backtest-results-container", className="mt-3")
                        )
                    ])
                ])
            ], id="tab-backtest"),
            
            # Tab 3: Yield Curve Statistics & Analysis -------------------------
            dbc.Tab(label="Statistics & Analysis", tab_id="tab-analysis", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Yield Curve Analysis Tools", className="mt-3"),
                        # Subtabs for different analysis types
                        dbc.Tabs([
                            # Statistical summary of yield curve data
                            dbc.Tab(label="Descriptive Statistics", children=[
                                html.Div(id="descriptive-stats-container", className="mt-3")
                            ]),
                            # Yield curve spread analysis (e.g., 2s10s, 3m10y)
                            dbc.Tab(label="Yield Curve Spreads", children=[
                                html.P("Analysis of key yield curve spreads and inversions", className="mt-3"),
                                html.Div(id="yield-spreads-container")
                            ]),
                            # Turning point analysis (peaks and troughs)
                            dbc.Tab(label="Turning Points", children=[
                                html.P("Identification of local maxima and minima in yield curves", className="mt-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Select Maturity:"),
                                        dcc.Dropdown(
                                            id="turning-points-maturity-dropdown",
                                            options=[{"label": f"{int(m)}Y" if m >= 1 else f"{int(round(m*12))}M", "value": m}
                                                     for m in sorted(config.MATURITIES_LIST, reverse=True)],
                                            value=10  # Default to 10Y benchmark
                                        )
                                    ], width=6)
                                ]),
                                html.Div(id="turning-points-container", className="mt-3")
                            ]),
                            # VIX correlation analysis (disabled if VIX data unavailable)
                            dbc.Tab(label="VIX Correlations", disabled=vix_data is None or vix_data.empty, children=[
                                html.P("Analysis of correlations between VIX and yield curves", className="mt-3"),
                                html.Div(id="vix-correlations-container")
                            ])
                        ])
                    ])
                ])
            ], id="tab-analysis")
        ], id="main-tabs")
    ])

    # ----------------------
    # Callback Functions
    # ----------------------

    @app.callback(
        [Output("yield-curve-graph", "figure"),
        Output("current-model-store", "data"),
        Output("selected-maturities-store", "data")],
        [Input("forecast-model-dropdown", "value"),
        Input("maturities-dropdown", "value")],
        [State("current-model-store", "data"),
        State("selected-maturities-store", "data")],
        prevent_initial_call=False
    )
    def update_figure(forecast_model, selected_maturities, current_model, stored_maturities):
        """Updates the 3D yield curve visualization based on user selections.
        
        Purpose:
        Generates a new 3D surface plot when the user changes the forecast model
        or selected maturities. Preserves state for persistent settings.
        
        Args:
            forecast_model: Selected forecasting model name.
            selected_maturities: List of maturities to highlight.
            current_model: Currently stored model name (fallback).
            stored_maturities: Currently stored maturities (fallback).
            
        Returns:
            tuple: (figure object, updated model, updated maturities)
        """
        # Use fallback values if inputs are empty/None
        if forecast_model is None or forecast_model == "":
            forecast_model = current_model or "Heuristic"
        
        if selected_maturities is None:
            selected_maturities = stored_maturities or []
        
        # Generate the 3D visualization with error handling
        try:
            fig = plotting.generate_surface_plot(
                forecast_model, 
                selected_maturities, 
                config.DB_FILE, 
                config.DEFAULT_START_DATE, 
                config.DEFAULT_END_DATE, 
                vix_data
            )
            return fig, forecast_model, selected_maturities
        except Exception as e:
            print(f"Error in update_figure: {str(e)}")
            # Create error message figure for user feedback
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating visualization: {str(e)}. Please refresh the page.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            # Return error figure but preserve current state
            return fig, forecast_model, selected_maturities

    @app.callback(
        Output("yield-curve-graph", "figure", allow_duplicate=True),
        [Input("main-tabs", "active_tab"),
        Input("viz-needs-refresh", "data")],
        [State("current-model-store", "data"),
        State("selected-maturities-store", "data")],
        prevent_initial_call=True
    )
    def refresh_visualization_on_tab_change(active_tab, needs_refresh, current_model, selected_maturities):
        """Refreshes the visualization when returning to the visualization tab.
        
        Purpose:
        This ensures visualization is updated after model changes or backtest runs.
        Also frees memory through garbage collection to prevent memory leaks.
        
        Args:
            active_tab: Currently active tab ID.
            needs_refresh: Flag indicating a refresh is needed.
            current_model: Currently selected model.
            selected_maturities: Currently selected maturities.
            
        Returns:
            go.Figure: Updated figure object or PreventUpdate exception.
        """
        # Only refresh when returning to visualization tab or explicitly triggered
        if active_tab == "tab-visualization" or needs_refresh:
            # Force garbage collection to free memory
            gc.collect()
            try:
                return plotting.generate_surface_plot(
                    current_model, 
                    selected_maturities, 
                    config.DB_FILE, 
                    config.DEFAULT_START_DATE, 
                    config.DEFAULT_END_DATE, 
                    vix_data
                )
            except Exception as e:
                print(f"Error refreshing visualization: {str(e)}")
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error refreshing visualization: {str(e)}. Please refresh the page.",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
                return fig
        
        # Don't update for other tab changes to avoid unnecessary computations
        raise PreventUpdate

    @app.callback(
        Output("backtest-maturities-dropdown", "value"),
        Input("backtest-maturities-dropdown", "value"),
        prevent_initial_call=True
    )
    def handle_select_all_maturities(selected_values):
        """Handles the 'Select All' option in the backtest maturities dropdown.
        
        Purpose:
        Special case handling for the 'all' value, which expands to all available maturities.
        
        Args:
            selected_values: List of currently selected values.
            
        Returns:
            list: Either all maturities or the unchanged selection.
        """
        if selected_values and "all" in selected_values:
            # Return full list of maturities when "all" is selected
            return config.MATURITIES_LIST
        # Otherwise return selection unchanged
        return selected_values

    # Note: Additional callbacks would be defined here for:
    # - Running backtests (backtest-results-container)
    # - Updating descriptive statistics (descriptive-stats-container)
    # - Updating yield spreads analysis (yield-spreads-container)
    # - Updating turning points analysis (turning-points-container)
    # - Updating VIX correlations (vix-correlations-container)
    
    return app

# Create the Dash application
app = create_app()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)  # Enable debug mode for development