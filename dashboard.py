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
            # Tab 1: 3D Yield Curve Visualization --------------------------------------------------
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
            
            # Tab 2: Backtesting & Model Evaluation --------------------------------------------
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
            
            # Tab 3: Yield Curve Statistics & Analysis ---------------------------------------------
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

    # ------------------------------------------------------------
    #                     Callback Functions
    # ------------------------------------------------------------

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

    @app.callback(
        [Output("backtest-results-container", "children"),
         Output("viz-needs-refresh", "data")],
        Input("run-backtest-btn", "n_clicks"),
        [State("backtest-maturities-dropdown", "value"),
         State("forecast-horizons-dropdown", "value"),
         State("backtest-model-dropdown", "value")],
        prevent_initial_call=True
    )
    def run_backtest_analysis(n_clicks, selected_maturities, forecast_horizons, forecast_model):
        """Executes backtest analysis and displays results.
        
        Purpose:
        Runs the backtest for selected maturities across specified forecast horizons
        using the chosen model. Displays detailed metrics and visualization of results.
        
        Args:
            n_clicks: Number of button clicks (not used except for triggering).
            selected_maturities: List of maturities to backtest.
            forecast_horizons: List of forecast horizons in months.
            forecast_model: Selected forecasting model.
            
        Returns:
            tuple: (results UI component, refresh visualization flag)
        """
        if not n_clicks or not selected_maturities or not forecast_horizons:
            return html.Div("Please select maturities and forecast horizons to run backtest."), False
        
        # Create deep copy of maturities list to avoid state conflicts
        backtest_maturities = list(selected_maturities).copy() if selected_maturities else []
        
        try:
            # Run the backtest - with isolation for memory management
            # Force garbage collection before running to ensure maximum memory availability
            gc.collect()
            
            backtest_results = analysis.run_backtest(
                config.DB_FILE, 
                config.DEFAULT_START_DATE, 
                config.DEFAULT_END_DATE, 
                backtest_maturities, 
                forecast_horizons=forecast_horizons,
                model=forecast_model,
                vix_data=vix_data
            )
            
            if not backtest_results:
                return html.Div("No backtest results available. Please check your data."), True
            
            children = []
            
            # Show how many backtest points were used
            sample_maturity = next(iter(backtest_results.keys()))
            sample_horizon = f"{forecast_horizons[0]}m_data"
            num_backtest_points = 0
            if sample_maturity in backtest_results and sample_horizon in backtest_results[sample_maturity]:
                num_backtest_points = len(backtest_results[sample_maturity][sample_horizon]['dates'])
            
            children.append(html.P(f"Backtest performed using {num_backtest_points} historical points with the {forecast_model} model.", className="mt-3"))
            
            # Create a compact single table with maturity as rows and metrics grouped by forecast horizon as columns
            table_data = []
            
            # Define thresholds for color coding based on metric types
            # Lower is better for MSE, RMSE, MAE, MAPE; Higher is better for R2 and Directional_Accuracy
            thresholds = {
                'MSE': {
                    'excellent': 0.02,   # Green
                    'good': 0.05,        # Light green
                    'moderate': 0.1,     # Yellow
                    'poor': 0.2          # Above this is red
                },
                'RMSE': {
                    'excellent': 0.1,    # Green
                    'good': 0.2,         # Light green
                    'moderate': 0.3,     # Yellow
                    'poor': 0.5          # Above this is red
                },
                'MAE': {
                    'excellent': 0.1,    # Green
                    'good': 0.2,         # Light green
                    'moderate': 0.3,     # Yellow
                    'poor': 0.5          # Above this is red
                },
                'MAPE': {
                    'excellent': 5,      # Green
                    'good': 10,          # Light green
                    'moderate': 15,      # Yellow
                    'poor': 25           # Above this is red
                },
                'R2': {
                    'excellent': 0.85,   # Green
                    'good': 0.7,         # Light green
                    'moderate': 0.5,     # Yellow
                    'poor': 0.3          # Below this is red
                },
                'Directional_Accuracy': {
                    'excellent': 0.75,   # Green
                    'good': 0.65,        # Light green
                    'moderate': 0.55,    # Yellow
                    'poor': 0.5          # Below this is red (worse than a coin flip)
                }
            }
            
            # Collect all values to calculate metric-specific ranges
            metric_values = {metric: [] for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']}
            
            for maturity in selected_maturities:
                for horizon in forecast_horizons:
                    for metric in metric_values.keys():
                        key = f"{horizon}m_{metric}"
                        if maturity in backtest_results and key in backtest_results[maturity] and pd.notnull(backtest_results[maturity][key]):
                            metric_values[metric].append(backtest_results[maturity][key])
            
            # Define color style functions based on metric type and value
            def get_cell_style(metric, value):
                """Returns appropriate cell style based on metric value quality."""
                if pd.isna(value) or value == '-':
                    return {}
                
                try:
                    # Extract numeric value if string representation
                    if isinstance(value, str):
                        if value.endswith('%'):
                            value = float(value.replace('%', ''))
                        else:
                            value = float(value)
                            
                    # For R2 and Directional_Accuracy, higher is better
                    if metric in ['R2', 'Directional_Accuracy']:
                        if value >= thresholds[metric]['excellent']: 
                            return {'backgroundColor': '#1d8936', 'color': 'white'}  # Green
                        elif value >= thresholds[metric]['good']: 
                            return {'backgroundColor': '#5cb85c', 'color': 'white'}  # Light green
                        elif value >= thresholds[metric]['moderate']: 
                            return {'backgroundColor': '#ffc107', 'color': 'black'}  # Yellow
                        elif value >= thresholds[metric]['poor']: 
                            return {'backgroundColor': '#fd7e14', 'color': 'white'}  # Orange
                        else: 
                            return {'backgroundColor': '#dc3545', 'color': 'white'}  # Red
                    # For other metrics, lower is better
                    else:
                        if value <= thresholds[metric]['excellent']: 
                            return {'backgroundColor': '#1d8936', 'color': 'white'}  # Green
                        elif value <= thresholds[metric]['good']: 
                            return {'backgroundColor': '#5cb85c', 'color': 'white'}  # Light green
                        elif value <= thresholds[metric]['moderate']: 
                            return {'backgroundColor': '#ffc107', 'color': 'black'}  # Yellow
                        elif value <= thresholds[metric]['poor']: 
                            return {'backgroundColor': '#fd7e14', 'color': 'white'}  # Orange
                        else: 
                            return {'backgroundColor': '#dc3545', 'color': 'white'}  # Red
                except:
                    return {}
            
            # Prepare display data for each maturity
            for maturity in sorted(selected_maturities, reverse=True):
                maturity_label = f"{int(maturity)}Y" if maturity >= 1 else f"{int(round(maturity*12))}M"
                row = {'Maturity': maturity_label}
                row_styles = {'Maturity': {}}
                
                # Add data for each forecast horizon and metric
                for horizon in sorted(forecast_horizons):
                    for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']:
                        key = f"{horizon}m_{metric}"
                        col_key = f"{horizon}m_{metric}"
                        
                        if key in backtest_results[maturity]:
                            value = backtest_results[maturity][key]
                            # Format values appropriately by metric type
                            if metric in ['MAPE', 'Directional_Accuracy']:
                                formatted_value = f"{value:.2f}%" if pd.notnull(value) else "-"
                            elif metric in ['MSE', 'RMSE']:
                                formatted_value = f"{value:.4f}" if pd.notnull(value) else "-"
                            else:
                                formatted_value = f"{value:.3f}" if pd.notnull(value) else "-"
                            
                            row[col_key] = formatted_value
                            row_styles[col_key] = get_cell_style(metric, value)
                        else:
                            row[col_key] = "-"
                            row_styles[col_key] = {}
                
                table_data.append((row, row_styles))
            
            # Create a custom HTML table with header groups and color coding
            header_html = html.Thead([
                # First row - Forecast horizons
                html.Tr([
                    html.Th("Maturity", rowSpan=2, className="align-middle"),
                    *[html.Th(f"{horizon}-Month Forecast", colSpan=6, className="text-center") 
                      for horizon in sorted(forecast_horizons)]
                ]),
                # Second row - Metrics
                html.Tr([
                    *[html.Th(metric, className="text-center") 
                      for horizon in sorted(forecast_horizons) 
                      for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']]
                ])
            ])
            
            # Create table body with color-coded cells
            rows = []
            for row_data, row_styles in table_data:
                cells = [html.Td(row_data['Maturity'], className="font-weight-bold", style=row_styles['Maturity'])]
                for horizon in sorted(forecast_horizons):
                    for metric in ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']:
                        key = f"{horizon}m_{metric}"
                        cells.append(html.Td(row_data[key], className="text-center", style=row_styles[key]))
                rows.append(html.Tr(cells))
            
            body_html = html.Tbody(rows)
            
            # Complete table using dbc.Table
            table_html = dbc.Table([header_html, body_html], 
                                  bordered=True,
                                  hover=True,
                                  responsive=True,
                                  className="metrics-table")
            
            children.append(html.H5("Backtest Metrics Summary", className="mt-4"))
            
            # Add legend for color coding with inline styles
            legend_style = {
                "display": "flex", 
                "flexWrap": "wrap", 
                "marginBottom": "15px"
            }
            
            legend_item_style = {
                "display": "flex", 
                "alignItems": "center", 
                "marginRight": "20px"
            }
            
            legend_box_style = {
                "width": "20px", 
                "height": "20px", 
                "marginRight": "5px"
            }
            
            legend_items = [
                html.Div([
                    html.Div(style={**legend_box_style, "backgroundColor": "#1d8936"}),
                    html.Span("Excellent")
                ], style=legend_item_style),
                html.Div([
                    html.Div(style={**legend_box_style, "backgroundColor": "#5cb85c"}),
                    html.Span("Good")
                ], style=legend_item_style),
                html.Div([
                    html.Div(style={**legend_box_style, "backgroundColor": "#ffc107"}),
                    html.Span("Moderate")
                ], style=legend_item_style),
                html.Div([
                    html.Div(style={**legend_box_style, "backgroundColor": "#fd7e14"}),
                    html.Span("Poor")
                ], style=legend_item_style),
                html.Div([
                    html.Div(style={**legend_box_style, "backgroundColor": "#dc3545"}),
                    html.Span("Very Poor")
                ], style=legend_item_style)
            ]
            
            legend_html = html.Div(legend_items, style=legend_style)
            
            children.append(legend_html)
            children.append(table_html)
            
            # Create visualizations for each maturity
            for maturity in selected_maturities:
                maturity_label = f"{int(maturity)}Y" if maturity >= 1 else f"{int(round(maturity*12))}M"
                
                # Create figure with subplot for each forecast horizon
                fig = make_subplots(
                    rows=len(forecast_horizons), 
                    cols=1,
                    subplot_titles=[f"{horizon}-Month Forecast Backtest for {maturity_label}" for horizon in forecast_horizons],
                    vertical_spacing=0.1
                )
                
                for i, horizon in enumerate(forecast_horizons):
                    data_key = f"{horizon}m_data"
                    if data_key in backtest_results[maturity]:
                        backtest_data = backtest_results[maturity][data_key]
                        
                        if not backtest_data['dates']:
                            continue
                        
                        # Plot actual values
                        fig.add_trace(
                            go.Scatter(
                                x=backtest_data['dates'],
                                y=backtest_data['actuals'],
                                mode='lines+markers',
                                name=f'Actual {maturity_label}',
                                line=dict(color='blue'),
                                showlegend=i == 0
                            ),
                            row=i+1, col=1
                        )
                        
                        # Plot forecast values
                        fig.add_trace(
                            go.Scatter(
                                x=backtest_data['dates'],
                                y=backtest_data['forecasts'],
                                mode='lines+markers',
                                name=f'Forecast {maturity_label}',
                                line=dict(color='red', dash='dash'),
                                showlegend=i == 0
                            ),
                            row=i+1, col=1
                        )
                        
                        # Add error bands
                        errors = [a - f for a, f in zip(backtest_data['actuals'], backtest_data['forecasts'])]
                        
                        fig.add_trace(
                            go.Scatter(
                                x=backtest_data['dates'],
                                y=errors,
                                mode='lines+markers',
                                name=f'Error',
                                line=dict(color='green'),
                                showlegend=i == 0
                            ),
                            row=i+1, col=1
                        )
                
                fig.update_layout(
                    height=300 * len(forecast_horizons),
                    title_text=f"Backtest Results for {maturity_label} ({forecast_model} model)",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                children.append(dcc.Graph(figure=fig, className="mt-4"))
            
            # Force garbage collection at the end to clean up memory
            gc.collect()
            
            # Return the results UI and flag to refresh visualization
            return html.Div(children), True
            
        except Exception as e:
            error_message = html.Div([
                html.H4("Error running backtest", className="text-danger"),
                html.P(f"Error details: {str(e)}")
            ])
            return error_message, True

    @app.callback(
        Output("descriptive-stats-container", "children"),
        [Input("backtest-maturities-dropdown", "value"),
         Input("main-tabs", "active_tab")],
        prevent_initial_call=True
    )
    def update_descriptive_stats(selected_maturities, active_tab):
        """Updates the descriptive statistics panel with yield curve data analysis.
        
        Purpose:
        Generates comprehensive statistical summary when the Statistics tab is active
        and maturities are selected.
        
        Args:
            selected_maturities: List of maturities to analyze.
            active_tab: Currently active tab ID.
            
        Returns:
            html.Div: Container with statistical analysis components.
        """
        # Only update when on the analysis tab and maturities are selected
        if active_tab != "tab-analysis" or not selected_maturities:
            raise PreventUpdate
            
        # Generate statistical analysis
        stats_df = analysis.generate_descriptive_statistics(
            config.DB_FILE, 
            config.DEFAULT_START_DATE, 
            config.DEFAULT_END_DATE, 
            selected_maturities
        )
        
        if stats_df.empty:
            return html.Div("No statistics available. Please check your data.")
        
        # Format table for display
        display_df = stats_df.reset_index()
        display_df.columns = ['Maturity'] + list(display_df.columns[1:])
        display_df['Maturity'] = display_df['Maturity'].apply(
            lambda m: f"{int(m)}Y" if float(m) >= 1 else f"{int(round(float(m)*12))}M"
        )
        
        # Create heatmap for key metrics
        key_metrics = ['mean', 'median', 'std', 'min', 'max', 'first_obs', 'last_obs', 'total_change', 'recent_trend']
        heatmap_df = stats_df[key_metrics]
        
        fig = px.imshow(
            heatmap_df,
            labels=dict(x="Metric", y="Maturity", color="Value"),
            x=key_metrics,
            y=[f"{int(m)}Y" if float(m) >= 1 else f"{int(round(float(m)*12))}M" for m in heatmap_df.index],
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        
        fig.update_layout(
            title="Yield Curve Statistics Heatmap",
            height=400
        )
        
        # Create time series comparison chart for selected maturities
        hist_df = data.prepare_data(config.DB_FILE, config.DEFAULT_START_DATE, config.DEFAULT_END_DATE)
        hist_df.index = pd.to_datetime(hist_df.index)
        
        time_series_fig = go.Figure()
        
        for maturity in selected_maturities:
            if maturity in hist_df.columns:
                maturity_label = f"{int(maturity)}Y" if maturity >= 1 else f"{int(round(maturity*12))}M"
                time_series_fig.add_trace(
                    go.Scatter(
                        x=hist_df.index,
                        y=hist_df[maturity],
                        mode='lines',
                        name=maturity_label
                    )
                )
        
        time_series_fig.update_layout(
            title="Historical Yield Curve Time Series",
            xaxis_title="Date",
            yaxis_title="Yield (%)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Create additional statistical visualization - autocorrelation
        acf_fig = go.Figure()
        for maturity in selected_maturities:
            if maturity in hist_df.columns:
                # Calculate autocorrelation for this maturity (up to 12 lags)
                series = hist_df[maturity].dropna()
                acf_values = [series.autocorr(lag) for lag in range(1, 13)]
                
                maturity_label = f"{int(maturity)}Y" if maturity >= 1 else f"{int(round(maturity*12))}M"
                acf_fig.add_trace(
                    go.Bar(
                        x=list(range(1, 13)),
                        y=acf_values,
                        name=maturity_label
                    )
                )
        
        acf_fig.update_layout(
            title="Autocorrelation Function (1-12 Month Lags)",
            xaxis_title="Lag (Months)",
            yaxis_title="Autocorrelation",
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Create table
        table = dbc.Table.from_dataframe(
            display_df.round(4),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True
        )
        
        return html.Div([
            html.H5("Historical Yield Curves", className="mt-4"),
            dcc.Graph(figure=time_series_fig),
            html.H5("Descriptive Statistics Heatmap", className="mt-4"),
            dcc.Graph(figure=fig),
            html.H5("Autocorrelation Analysis", className="mt-4"),
            dcc.Graph(figure=acf_fig),
            html.H5("Detailed Statistics", className="mt-4"),
            table
        ])

    @app.callback(
        Output("yield-spreads-container", "children"),
        [Input("main-tabs", "active_tab")],
        prevent_initial_call=True
    )
    def update_yield_spreads(active_tab):
        """Updates the yield curve spreads analysis panel.
        
        Purpose:
        Generates visualizations of key yield curve spreads and inversions when
        the corresponding analysis tab is active.
        
        Args:
            active_tab: Currently active tab ID.
            
        Returns:
            html.Div: Container with yield spread analysis components.
        """
        # Only update when on the analysis tab
        if active_tab != "tab-analysis":
            raise PreventUpdate
            
        # Generate the spreads visualization
        fig = plotting.plot_yield_spread_trends(
            config.DB_FILE, 
            config.DEFAULT_START_DATE, 
            config.DEFAULT_END_DATE
        )
        
        # Define standard yield curve spreads to analyze
        maturity_pairs = [
            (10, 2),     # 10Y-2Y - classic recession indicator
            (10, 0.25),  # 10Y-3M - alternative recession indicator
            (30, 5),     # 30Y-5Y - long term expectations
            (5, 2),      # 5Y-2Y - medium term expectations
            (2, 0.25)    # 2Y-3M - near term monetary policy expectations
        ]
        
        spreads_df = utils.calculate_yield_curve_differentials(
            config.DB_FILE, 
            config.DEFAULT_START_DATE, 
            config.DEFAULT_END_DATE, 
            maturity_pairs
        )
        
        if spreads_df.empty:
            return html.Div("No spread data available. Please check your data.")
        
        # Create recession probability model
        recession_prob_fig = go.Figure()
        
        # Check for 10Y-3M spread specifically
        spread_column = None
        for col in spreads_df.columns:
            if col == "10Y-0.25Y" or col == "10Y-3M":
                spread_column = col
                break
        
        if spread_column:
            # Get the spread data and ensure it's clean
            spread_values = spreads_df[spread_column].dropna()
            
            # Calculate simple logistic probability
            # P(recession) = 1 / (1 + exp(beta * spread))
            # where beta is calibrated so that spread of -0.5 gives ~80% probability
            beta = 5  # Calibration parameter
            recession_prob = 1 / (1 + np.exp(beta * spread_values))
            
            recession_prob_fig.add_trace(
                go.Scatter(
                    x=spread_values.index,
                    y=recession_prob * 100,  # Convert to percentage
                    mode='lines',
                    name='Recession Probability',
                    line=dict(color='red')
                )
            )
            
            # Add reference lines
            recession_prob_fig.add_shape(
                type="line",
                x0=spread_values.index.min(),
                y0=50,
                x1=spread_values.index.max(),
                y1=50,
                line=dict(color="orange", width=1, dash="dash")
            )
            
            recession_prob_fig.add_shape(
                type="line",
                x0=spread_values.index.min(),
                y0=70,
                x1=spread_values.index.max(),
                y1=70,
                line=dict(color="red", width=1, dash="dash")
            )
            
            recession_prob_fig.update_layout(
                title="Estimated 12-Month Recession Probability (Based on 10Y-3M Spread)",
                xaxis_title="Date",
                yaxis_title="Probability (%)",
                height=400,
                yaxis=dict(range=[0, 100])
            )
        else:
            # Create an empty figure with a message if spread is not available
            recession_prob_fig.add_annotation(
                text="10Y-3M spread data not available for recession probability model",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            recession_prob_fig.update_layout(height=400)
        
        # Create inversion heatmap
        inversion_cols = [col for col in spreads_df.columns if col.endswith('_inverted')]
        if inversion_cols:
            inversion_df = spreads_df[inversion_cols].astype(int)
            inversion_df.columns = [col.replace('_inverted', '') for col in inversion_cols]
            
            # Resample to monthly for better visualization
            monthly_inversions = inversion_df.resample('M').mean()
            
            inversion_fig = px.imshow(
                monthly_inversions.T,
                labels=dict(x="Date", y="Spread", color="Inverted"),
                color_continuous_scale=[[0, "white"], [1, "red"]],
                aspect="auto"
            )
            
            inversion_fig.update_layout(
                title="Yield Curve Inversion Periods",
                height=300
            )
        else:
            inversion_fig = go.Figure()
            inversion_fig.add_annotation(
                text="No inversion data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False
            )
            inversion_fig.update_layout(height=300)
        
        # Create recent spreads table
        recent_spreads = spreads_df.iloc[-12:].resample('M').last()
        recent_spread_cols = [col for col in recent_spreads.columns if not col.endswith('_inverted')]
        recent_spreads_table = dbc.Table.from_dataframe(
            recent_spreads[recent_spread_cols].reset_index().round(4),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True
        )
        
        # Create summary metrics
        summary_metrics = []
        for col in [col for col in spreads_df.columns if not col.endswith('_inverted')]:
            inversion_col = f"{col}_inverted"
            if inversion_col in spreads_df.columns:
                # Calculate inversion statistics
                inversion_count = spreads_df[inversion_col].sum()
                inversion_pct = (inversion_count / len(spreads_df)) * 100
                
                # Find current streak (consecutive months of inversion or non-inversion)
                current_state = spreads_df[inversion_col].iloc[-1]
                streak = 0
                for i in range(len(spreads_df) - 1, -1, -1):
                    if spreads_df[inversion_col].iloc[i] == current_state:
                        streak += 1
                    else:
                        break
                
                streak_state = "inverted" if current_state else "non-inverted"
                
                # Most recent value
                current_spread = spreads_df[col].iloc[-1]
                
                summary_metrics.append({
                    "Spread": col,
                    "Current Value": round(current_spread, 4),
                    "Currently Inverted": "Yes" if current_spread < 0 else "No",
                    "Inversion Months": inversion_count,
                    "Inversion %": round(inversion_pct, 2),
                    f"Current Streak ({streak_state})": streak
                })
        
        if summary_metrics:
            summary_df = pd.DataFrame(summary_metrics)
            summary_table = dbc.Table.from_dataframe(
                summary_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
        else:
            summary_table = html.Div("No summary metrics available.")
        
        return html.Div([
            html.H5("Yield Curve Spread Trends", className="mt-3"),
            dcc.Graph(figure=fig),
            html.H5("Yield Curve Inversion Periods", className="mt-4"),
            dcc.Graph(figure=inversion_fig),
            html.H5("Recession Probability Model", className="mt-4"),
            dcc.Graph(figure=recession_prob_fig),
            html.H5("Spread Summary Statistics", className="mt-4"),
            summary_table,
            html.H5("Recent Monthly Spreads", className="mt-4"),
            recent_spreads_table
        ])

    @app.callback(
        Output("turning-points-container", "children"),
        [Input("turning-points-maturity-dropdown", "value"),
         Input("main-tabs", "active_tab")],
        prevent_initial_call=True
    )
    def update_turning_points(selected_maturity, active_tab):
        """Updates the turning points analysis panel.
        
        Purpose:
        Identifies and visualizes local maxima and minima in the yield curve history
        for a selected maturity.
        
        Args:
            selected_maturity: Maturity to analyze for turning points.
            active_tab: Currently active tab ID.
            
        Returns:
            html.Div: Container with turning points analysis components.
        """
        # Only update when on the analysis tab and a maturity is selected
        if active_tab != "tab-analysis" or not selected_maturity:
            raise PreventUpdate
            
        # Generate the turning points visualization
        fig = plotting.plot_turning_points(
            config.DB_FILE, 
            config.DEFAULT_START_DATE, 
            config.DEFAULT_END_DATE, 
            selected_maturity
        )
        
        # Get turning points data
        turning_points = analysis.analyze_turning_points(
            config.DB_FILE, 
            config.DEFAULT_START_DATE, 
            config.DEFAULT_END_DATE, 
            [selected_maturity]
        )
        
        if not turning_points or selected_maturity not in turning_points:
            return html.Div("No turning points found for selected maturity.")
        
        # Create table of turning points
        if turning_points[selected_maturity]:
            tp_data = []
            for tp in turning_points[selected_maturity]:
                tp_data.append({
                    "Date": tp['date'].strftime("%Y-%m-%d"),
                    "Type": tp['type'].capitalize(),
                    "Yield": round(tp['value'], 4)
                })
            
            tp_df = pd.DataFrame(tp_data)
            tp_df = tp_df.sort_values("Date", ascending=False)
            
            tp_table = dbc.Table.from_dataframe(
                tp_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
        else:
            tp_table = html.Div("No turning points found.")
        
        # Calculate statistics on cycles
        peaks = [tp for tp in turning_points[selected_maturity] if tp['type'] == 'peak']
        troughs = [tp for tp in turning_points[selected_maturity] if tp['type'] == 'trough']
        
        if len(peaks) > 1 and len(troughs) > 1:
            # Sort all turning points by date
            all_tp = peaks + troughs
            all_tp.sort(key=lambda x: x['date'])
            
            # Calculate cycle lengths
            cycle_data = []
            
            # Peak-to-peak cycles
            if len(peaks) >= 2:
                peak_dates = [p['date'] for p in peaks]
                peak_dates.sort()
                
                for i in range(1, len(peak_dates)):
                    days = (peak_dates[i] - peak_dates[i-1]).days
                    cycle_data.append({
                        "Cycle Type": "Peak-to-Peak",
                        "Start Date": peak_dates[i-1].strftime("%Y-%m-%d"),
                        "End Date": peak_dates[i].strftime("%Y-%m-%d"),
                        "Duration (Days)": days,
                        "Duration (Months)": round(days / 30.44, 1)  # Approximate months
                    })
            
            # Trough-to-trough cycles
            if len(troughs) >= 2:
                trough_dates = [t['date'] for t in troughs]
                trough_dates.sort()
                
                for i in range(1, len(trough_dates)):
                    days = (trough_dates[i] - trough_dates[i-1]).days
                    cycle_data.append({
                        "Cycle Type": "Trough-to-Trough",
                        "Start Date": trough_dates[i-1].strftime("%Y-%m-%d"),
                        "End Date": trough_dates[i].strftime("%Y-%m-%d"),
                        "Duration (Days)": days,
                        "Duration (Months)": round(days / 30.44, 1)  # Approximate months
                    })
            
            if cycle_data:
                cycle_df = pd.DataFrame(cycle_data)
                cycle_table = dbc.Table.from_dataframe(
                    cycle_df,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True
                )
                
                # Calculate summary statistics
                p2p_cycles = [row for row in cycle_data if row["Cycle Type"] == "Peak-to-Peak"]
                t2t_cycles = [row for row in cycle_data if row["Cycle Type"] == "Trough-to-Trough"]
                
                p2p_avg = sum(c["Duration (Months)"] for c in p2p_cycles) / len(p2p_cycles) if p2p_cycles else 0
                t2t_avg = sum(c["Duration (Months)"] for c in t2t_cycles) / len(t2t_cycles) if t2t_cycles else 0
                
                summary_text = [
                    html.P(f"Average Peak-to-Peak Cycle: {p2p_avg:.1f} months ({p2p_avg/12:.1f} years)"),
                    html.P(f"Average Trough-to-Trough Cycle: {t2t_avg:.1f} months ({t2t_avg/12:.1f} years)")
                ]
            else:
                cycle_table = html.Div("No cycle data available.")
                summary_text = []
        else:
            cycle_table = html.Div("Not enough turning points to analyze cycles.")
            summary_text = []
        
        # Rate of change analysis
        # Get historical data for this maturity
        hist_df = data.prepare_data(config.DB_FILE, config.DEFAULT_START_DATE, config.DEFAULT_END_DATE)
        roc_fig = go.Figure()
        
        if selected_maturity in hist_df.columns:
            # Calculate 3-month rate of change
            series = hist_df[selected_maturity]
            roc = series.pct_change(3) * 100  # 3-month percentage change
            
            roc_fig.add_trace(
                go.Scatter(
                    x=roc.index,
                    y=roc,
                    mode='lines',
                    name='3-Month Rate of Change',
                    line=dict(color='blue')
                )
            )
            
            # Add zero line
            roc_fig.add_shape(
                type="line",
                x0=roc.index.min(),
                y0=0,
                x1=roc.index.max(),
                y1=0,
                line=dict(color="red", width=1, dash="dash")
            )
            
            maturity_label = f"{int(selected_maturity)}Y" if selected_maturity >= 1 else f"{int(round(selected_maturity*12))}M"
            roc_fig.update_layout(
                title=f"Rate of Change Analysis for {maturity_label} Yield (3-Month)",
                xaxis_title="Date",
                yaxis_title="3-Month Rate of Change (%)",
                height=400
            )
        
        return html.Div([
            dcc.Graph(figure=fig),
            html.Div(summary_text),
            html.H5("Rate of Change Analysis", className="mt-4"),
            dcc.Graph(figure=roc_fig),
            html.H5("Turning Points", className="mt-4"),
            tp_table,
            html.H5("Yield Cycles", className="mt-4"),
            cycle_table
        ])

    @app.callback(
        Output("vix-correlations-container", "children"),
        [Input("backtest-maturities-dropdown", "value"),
         Input("main-tabs", "active_tab")],
        prevent_initial_call=True
    )
    def update_vix_correlations(selected_maturities, active_tab):
        """Updates the VIX correlation analysis panel.
        
        Purpose:
        Analyzes correlations between market volatility (VIX) and Treasury yields
        across different maturity points and volatility regimes.
        
        Args:
            selected_maturities: List of maturities to analyze.
            active_tab: Currently active tab ID.
            
        Returns:
            html.Div: Container with VIX correlation analysis components.
        """
        # Only update when on the analysis tab and when VIX data is available
        if active_tab != "tab-analysis" or not selected_maturities or vix_data is None or vix_data.empty:
            raise PreventUpdate
        
        # Calculate correlations between VIX and yield curve
        correlations_df = analysis.calculate_vix_yield_correlations(
            config.DB_FILE,
            config.DEFAULT_START_DATE,
            config.DEFAULT_END_DATE,
            selected_maturities,
            vix_data
        )
        
        if correlations_df.empty:
            return html.Div("Unable to calculate VIX correlations. Please check your data.")
        
        # Create heatmap of correlations
        corr_fig = px.imshow(
            correlations_df,
            labels=dict(x="Correlation Type", y="Maturity", color="Correlation"),
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        
        corr_fig.update_layout(
            title="VIX-Yield Curve Correlations",
            height=400
        )
        
        # Create VIX vs yields scatter plots for selected maturities
        scatter_plots = []
        
        # Prepare yield data
        hist_df = data.prepare_data(config.DB_FILE, config.DEFAULT_START_DATE, config.DEFAULT_END_DATE)
        
        # Prepare VIX data, aligning with yield dates
        vix_aligned = vix_data.set_index('date').reindex(hist_df.index, method='nearest')
        
        for maturity in selected_maturities:
            if maturity not in hist_df.columns:
                continue
                
            # Create scatter plot
            scatter_fig = px.scatter(
                x=vix_aligned['vix'],
                y=hist_df[maturity],
                trendline="ols",
                labels={
                    "x": "VIX Index",
                    "y": f"{int(maturity)}Y Yield" if maturity >= 1 else f"{int(round(maturity*12))}M Yield"
                }
            )
            
            scatter_fig.update_layout(
                title=f"VIX vs {int(maturity)}Y Yield" if maturity >= 1 else f"VIX vs {int(round(maturity*12))}M Yield",
                height=300
            )
            
            scatter_plots.append(dcc.Graph(figure=scatter_fig))
        
        # Create time series plot of VIX and selected yields
        ts_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add VIX
        ts_fig.add_trace(
            go.Scatter(
                x=vix_aligned.index,
                y=vix_aligned['vix'],
                name="VIX Index",
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        # Add yields
        for maturity in selected_maturities:
            if maturity not in hist_df.columns:
                continue
                
            maturity_label = f"{int(maturity)}Y" if maturity >= 1 else f"{int(round(maturity*12))}M"
            ts_fig.add_trace(
                go.Scatter(
                    x=hist_df.index,
                    y=hist_df[maturity],
                    name=f"{maturity_label} Yield",
                    line=dict(dash='dash')
                ),
                secondary_y=False
            )
        
        ts_fig.update_layout(
            title="VIX Index vs Treasury Yields Time Series",
            xaxis_title="Date",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        ts_fig.update_yaxes(title_text="Yield (%)", secondary_y=False)
        ts_fig.update_yaxes(title_text="VIX Index", secondary_y=True)
        
        # Create correlations table
        corr_table = dbc.Table.from_dataframe(
            correlations_df.reset_index().rename(columns={'index': 'Maturity'}),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True
        )
        
        # Create VIX regime analysis
        regime_fig = go.Figure()
        
        # Add VIX
        regime_fig.add_trace(
            go.Scatter(
                x=vix_aligned.index,
                y=vix_aligned['vix'],
                name="VIX Index",
                line=dict(color='black')
            )
        )
        
        # Add regime threshold lines
        regime_fig.add_shape(
            type="line",
            x0=vix_aligned.index.min(),
            y0=20,
            x1=vix_aligned.index.max(),
            y1=20,
            line=dict(color="orange", width=1, dash="dash")
        )
        
        regime_fig.add_shape(
            type="line",
            x0=vix_aligned.index.min(),
            y0=30,
            x1=vix_aligned.index.max(),
            y1=30,
            line=dict(color="red", width=1, dash="dash")
        )
        
        # Add color bands for different regimes
        regime_fig.add_hrect(
            y0=0, y1=20,
            fillcolor="green", opacity=0.1,
            layer="below", line_width=0,
        )
        
        regime_fig.add_hrect(
            y0=20, y1=30,
            fillcolor="orange", opacity=0.1,
            layer="below", line_width=0,
        )
        
        regime_fig.add_hrect(
            y0=30, y1=100,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0,
        )
        
        regime_fig.update_layout(
            title="VIX Volatility Regimes",
            xaxis_title="Date",
            yaxis_title="VIX Index",
            height=400,
            annotations=[
                dict(
                    x=0.02,
                    y=15,
                    xref="paper",
                    yref="y",
                    text="Low Volatility",
                    showarrow=False,
                    font=dict(color="green", size=12)
                ),
                dict(
                    x=0.02,
                    y=25,
                    xref="paper",
                    yref="y",
                    text="Medium Volatility",
                    showarrow=False,
                    font=dict(color="orange", size=12)
                ),
                dict(
                    x=0.02,
                    y=40,
                    xref="paper",
                    yref="y",
                    text="High Volatility",
                    showarrow=False,
                    font=dict(color="red", size=12)
                )
            ]
        )
        
        return html.Div([
            html.H5("VIX Volatility Regimes", className="mt-3"),
            dcc.Graph(figure=regime_fig),
            html.H5("VIX vs Treasury Yields", className="mt-4"),
            dcc.Graph(figure=ts_fig),
            html.H5("VIX-Yield Correlations Heatmap", className="mt-4"),
            dcc.Graph(figure=corr_fig),
            html.H5("Correlation Statistics", className="mt-4"),
            corr_table,
            html.H5("Scatter Plots", className="mt-4"),
            html.Div(scatter_plots)
        ])
    
    return app

# Create the Dash application
app = create_app()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)