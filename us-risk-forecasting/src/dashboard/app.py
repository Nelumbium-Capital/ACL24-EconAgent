"""
Interactive Risk Dashboard using Dash and Plotly.
Professional-grade visualization for financial risk metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.fred_client import FREDClient
from src.data.pipeline import DataPipeline, MissingValueHandler, FrequencyAligner
from src.data.data_models import SeriesConfig
from src.kri.calculator import KRICalculator
from src.kri.definitions import kri_registry, RiskLevel
from src.models.llm_forecaster import LLMEnsembleForecaster
from src.models.arima_forecaster import ARIMAForecaster
from src.models.ets_forecaster import ETSForecaster
from src.models.lstm_forecaster import LSTMForecaster
from src.simulation.model import RiskSimulationModel
from src.simulation.scenarios import get_scenario, SCENARIO_LIBRARY
from src.utils.logging_config import logger
from config import settings

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="Risk Forecasting Dashboard",
    update_title="Loading...",
    suppress_callback_exceptions=True
)

# Global data cache
data_cache = {
    'economic_data': None,
    'forecasts': None,
    'model_forecasts': None,  # Individual model forecasts for comparison
    'kris': None,
    'risk_levels': None,
    'scenario_results': None,  # Results from different scenarios
    'last_update': None
}


def fetch_and_process_data():
    """Fetch and process all data for dashboard."""
    logger.info("Fetching data for dashboard...")
    
    # Initialize components
    fred_client = FREDClient()
    pipeline = DataPipeline(fred_client)
    pipeline.add_transformer(MissingValueHandler(method='ffill'))
    pipeline.add_transformer(FrequencyAligner(target_frequency='M'))
    
    # Configure series
    series_config = {
        'unemployment': SeriesConfig(
            series_id='UNRATE',
            name='Unemployment Rate',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        ),
        'inflation': SeriesConfig(
            series_id='CPIAUCSL',
            name='CPI Inflation',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly',
            transformation='pct_change'
        ),
        'interest_rate': SeriesConfig(
            series_id='FEDFUNDS',
            name='Federal Funds Rate',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        ),
        'credit_spread': SeriesConfig(
            series_id='BAA10Y',
            name='BAA-Treasury Spread',
            start_date='2018-01-01',
            end_date='2024-01-01',
            frequency='monthly'
        )
    }
    
    # Process data
    economic_data = pipeline.process(series_config)
    
    # Generate forecasts with multiple models
    forecast_horizon = 12
    forecasts_dict = {}
    model_forecasts_dict = {}  # Store individual model forecasts
    
    for col in economic_data.columns:
        # Get series with proper DatetimeIndex
        series = economic_data[col].dropna()
        model_forecasts_dict[col] = {}
        
        logger.info(f"Generating forecasts for {col}...")
        
        try:
            # ARIMA forecast
            arima_model = ARIMAForecaster(auto_order=False, order=(2, 1, 2))
            arima_model.fit(series)
            arima_result = arima_model.forecast(horizon=forecast_horizon)
            arima_pred = arima_result.point_forecast
            model_forecasts_dict[col]['ARIMA'] = arima_pred
            logger.info(f"  ARIMA forecast for {col}: {arima_pred[0]:.4f} to {arima_pred[-1]:.4f}")
            
            # ETS forecast
            ets_model = ETSForecaster(trend='add', seasonal=None)
            ets_model.fit(series)
            ets_result = ets_model.forecast(horizon=forecast_horizon)
            ets_pred = ets_result.point_forecast
            model_forecasts_dict[col]['ETS'] = ets_pred
            logger.info(f"  ETS forecast for {col}: {ets_pred[0]:.4f} to {ets_pred[-1]:.4f}")
            
            # LLM Ensemble forecast (uses multiple models including LSTM)
            try:
                llm_forecaster = LLMEnsembleForecaster()
                llm_result = llm_forecaster.forecast(
                    series=series.values,
                    horizon=forecast_horizon,
                    series_name=col,
                    use_llm=False  # Use statistical ensemble for speed
                )
                llm_pred = llm_result['ensemble']
                model_forecasts_dict[col]['LLM_Ensemble'] = llm_pred
                logger.info(f"  LLM Ensemble forecast for {col}: {llm_pred[0]:.4f} to {llm_pred[-1]:.4f}")
                
                # Use LLM ensemble as primary forecast
                ensemble_pred = llm_pred
            except Exception as llm_error:
                logger.warning(f"  LLM forecast failed for {col}: {llm_error}, using ARIMA/ETS average")
                # Fallback to simple average
                ensemble_pred = (arima_pred + ets_pred) / 2
            
            model_forecasts_dict[col]['Ensemble'] = ensemble_pred
            forecasts_dict[col] = ensemble_pred
            
        except Exception as e:
            logger.error(f"All forecasts failed for {col}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to last value
            last_value = series.iloc[-1]
            naive_pred = np.full(forecast_horizon, last_value)
            model_forecasts_dict[col]['ARIMA'] = naive_pred
            model_forecasts_dict[col]['ETS'] = naive_pred
            model_forecasts_dict[col]['Ensemble'] = naive_pred
            forecasts_dict[col] = naive_pred
    
    # Create forecast DataFrame
    forecast_dates = pd.date_range(
        start=economic_data.index[-1] + pd.DateOffset(months=1),
        periods=forecast_horizon,
        freq='ME'
    )
    forecasts_df = pd.DataFrame(forecasts_dict, index=forecast_dates)
    
    # Compute KRIs
    kri_calc = KRICalculator()
    combined_data = pd.concat([economic_data.tail(12), forecasts_df])
    kris = kri_calc.compute_all_kris(forecasts=combined_data)
    risk_levels = kri_calc.evaluate_thresholds(kris)
    
    # Run scenario analysis with economic context
    logger.info("Running scenario simulations with economic forecasts...")
    scenario_results = {}
    
    # Get average forecasted values for scenario calibration
    avg_unemployment = forecasts_df['unemployment'].mean()
    avg_inflation = forecasts_df['inflation'].mean() if 'inflation' in forecasts_df else 0.02
    avg_interest_rate = forecasts_df['interest_rate'].mean()
    avg_credit_spread = forecasts_df['credit_spread'].mean()
    
    logger.info(f"Economic forecast averages: U={avg_unemployment:.2f}%, I={avg_inflation*100:.2f}%, R={avg_interest_rate:.2f}%, CS={avg_credit_spread:.2f}%")
    
    for scenario_name in ['baseline', 'recession', 'rate_shock', 'credit_crisis']:
        try:
            scenario = get_scenario(scenario_name)
            
            # Use more banks and firms for better statistics
            sim_model = RiskSimulationModel(
                n_banks=10,
                n_firms=50,
                scenario=scenario,
                random_seed=42  # For reproducibility
            )
            
            # Set initial economic conditions from forecasts
            sim_model.unemployment_rate = avg_unemployment / 100  # Convert to decimal
            sim_model.interest_rate = avg_interest_rate / 100
            sim_model.credit_spread = avg_credit_spread / 100
            
            # Run longer simulation for better convergence
            results = sim_model.run_simulation(n_steps=100)
            
            # Compute KRIs for this scenario using both forecasts and simulation
            scenario_kris = kri_calc.compute_all_kris(
                forecasts=combined_data,
                simulation_results=results
            )
            
            # Calculate additional stress metrics
            default_rates = results['default_rate'].values
            stress_metrics = {
                'mean_default': default_rates.mean(),
                'max_default': default_rates.max(),
                'var_95': np.percentile(default_rates, 95),
                'cvar_95': default_rates[default_rates >= np.percentile(default_rates, 95)].mean()
            }
            
            scenario_results[scenario_name] = {
                'kris': scenario_kris,
                'simulation': results,
                'stress_metrics': stress_metrics
            }
            logger.info(f"Completed {scenario_name} scenario - Default rate: {stress_metrics['mean_default']*100:.2f}%")
        except Exception as e:
            logger.error(f"Scenario {scenario_name} failed: {e}")
            import traceback
            traceback.print_exc()
            scenario_results[scenario_name] = None
    
    # Update cache
    data_cache['economic_data'] = economic_data
    data_cache['forecasts'] = forecasts_df
    data_cache['model_forecasts'] = model_forecasts_dict
    data_cache['kris'] = kris
    data_cache['risk_levels'] = risk_levels
    data_cache['scenario_results'] = scenario_results
    data_cache['last_update'] = datetime.now()
    
    logger.info("Dashboard data updated successfully")
    return economic_data, forecasts_df, kris, risk_levels


# Define color scheme
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#ff9800',
    'danger': '#d62728',
    'info': '#17a2b8',
    'background': '#f8f9fa',
    'card': '#ffffff',
    'text': '#212529',
    'border': '#dee2e6'
}

RISK_COLORS = {
    'low': '#28a745',
    'medium': '#ffc107',
    'high': '#fd7e14',
    'critical': '#dc3545'
}


# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("Financial Risk Forecasting Dashboard", 
                   style={'color': 'white', 'margin': '0', 'fontSize': '28px'}),
            html.P("Real-time risk monitoring and forecasting", 
                  style={'color': '#e0e0e0', 'margin': '5px 0 0 0'}),
        ], style={'flex': '1'}),
        html.Div([
            html.Button('Refresh Data', id='refresh-button', 
                       style={
                           'backgroundColor': '#28a745',
                           'color': 'white',
                           'border': 'none',
                           'padding': '10px 20px',
                           'borderRadius': '5px',
                           'cursor': 'pointer',
                           'fontSize': '14px'
                       }),
            html.Div(id='last-update', style={'color': '#e0e0e0', 'marginTop': '5px', 'fontSize': '12px'})
        ])
    ], style={
        'backgroundColor': '#2c3e50',
        'padding': '20px 40px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Main content
    html.Div([
        # KRI Summary Cards
        html.Div(id='kri-cards', style={'marginBottom': '30px'}),
        
        # Economic Indicators
        html.Div([
            html.H2("Economic Indicators", style={'marginBottom': '20px', 'color': COLORS['text']}),
            dcc.Graph(id='economic-indicators-chart', style={'height': '400px'})
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Forecasts
        html.Div([
            html.H2("12-Month Forecasts", style={'marginBottom': '20px', 'color': COLORS['text']}),
            dcc.Graph(id='forecasts-chart', style={'height': '400px'})
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Model Comparison
        html.Div([
            html.H2("Model Forecast Comparison", style={'marginBottom': '20px', 'color': COLORS['text']}),
            html.Div([
                html.Label("Select Indicator:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='model-comparison-indicator',
                    options=[
                        {'label': 'Unemployment Rate', 'value': 'unemployment'},
                        {'label': 'Inflation', 'value': 'inflation'},
                        {'label': 'Interest Rate', 'value': 'interest_rate'},
                        {'label': 'Credit Spread', 'value': 'credit_spread'}
                    ],
                    value='unemployment',
                    style={'width': '300px'}
                )
            ], style={'marginBottom': '15px'}),
            dcc.Graph(id='model-comparison-chart', style={'height': '400px'})
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Scenario Analysis
        html.Div([
            html.H2("Scenario Analysis", style={'marginBottom': '20px', 'color': COLORS['text']}),
            html.Div([
                html.Label("Select Scenario:", style={'marginRight': '10px', 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='scenario-selector',
                    options=[
                        {'label': 'Baseline', 'value': 'baseline'},
                        {'label': 'Recession', 'value': 'recession'},
                        {'label': 'Interest Rate Shock', 'value': 'rate_shock'},
                        {'label': 'Credit Crisis', 'value': 'credit_crisis'}
                    ],
                    value='baseline',
                    style={'width': '300px'}
                )
            ], style={'marginBottom': '15px'}),
            dcc.Graph(id='scenario-chart', style={'height': '400px'})
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Scenario Comparison
        html.Div([
            html.H2("Multi-Scenario Comparison", style={'marginBottom': '20px', 'color': COLORS['text']}),
            html.Div([
                html.Div([
                    dcc.Graph(id='scenario-comparison-kris', style={'height': '400px'})
                ], style={'flex': '1', 'marginRight': '15px'}),
                html.Div([
                    dcc.Graph(id='scenario-comparison-distribution', style={'height': '400px'})
                ], style={'flex': '1', 'marginLeft': '15px'})
            ], style={'display': 'flex'})
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Scenario Risk Metrics
        html.Div([
            html.H2("Scenario-Conditional Risk Metrics", style={'marginBottom': '20px', 'color': COLORS['text']}),
            html.Div(id='scenario-risk-metrics')
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Risk Heatmap
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Risk Heatmap", style={'marginBottom': '20px', 'color': COLORS['text']}),
                    dcc.Graph(id='risk-heatmap', style={'height': '400px'})
                ], style={'flex': '1', 'marginRight': '15px'}),
                
                html.Div([
                    html.H2("Risk Distribution", style={'marginBottom': '20px', 'color': COLORS['text']}),
                    dcc.Graph(id='risk-distribution', style={'height': '400px'})
                ], style={'flex': '1', 'marginLeft': '15px'})
            ], style={'display': 'flex'})
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # Detailed KRI Table
        html.Div([
            html.Div([
                html.H2("Key Risk Indicators Detail", style={'marginBottom': '0', 'color': COLORS['text'], 'flex': '1'}),
                html.Div([
                    html.Button('Export CSV', id='export-csv-button', 
                               style={
                                   'backgroundColor': COLORS['primary'],
                                   'color': 'white',
                                   'border': 'none',
                                   'padding': '8px 16px',
                                   'borderRadius': '5px',
                                   'cursor': 'pointer',
                                   'marginRight': '10px'
                               }),
                    html.Button('Export Excel', id='export-excel-button',
                               style={
                                   'backgroundColor': COLORS['success'],
                                   'color': 'white',
                                   'border': 'none',
                                   'padding': '8px 16px',
                                   'borderRadius': '5px',
                                   'cursor': 'pointer',
                                   'marginRight': '10px'
                               }),
                    html.Button('Export JSON', id='export-json-button',
                               style={
                                   'backgroundColor': COLORS['info'],
                                   'color': 'white',
                                   'border': 'none',
                                   'padding': '8px 16px',
                                   'borderRadius': '5px',
                                   'cursor': 'pointer'
                               }),
                    dcc.Download(id='download-data')
                ])
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '20px'}),
            html.Div(id='kri-table')
        ], style={
            'backgroundColor': COLORS['card'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),
        
        # KRI Drill-down Modal
        html.Div([
            html.Div(id='kri-drilldown-content')
        ], id='kri-drilldown-modal', style={'display': 'none'})
    ], style={
        'padding': '30px 40px',
        'backgroundColor': COLORS['background'],
        'minHeight': '100vh'
    }),
    
    # Hidden div for data storage
    dcc.Store(id='data-store'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)  # Update every minute
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0', 'padding': '0'})


@app.callback(
    [Output('data-store', 'data'),
     Output('last-update', 'children')],
    [Input('refresh-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_data(n_clicks, n_intervals):
    """Update data from FRED and recompute metrics."""
    try:
        fetch_and_process_data()
        update_time = data_cache['last_update'].strftime('%Y-%m-%d %H:%M:%S')
        return {'updated': True}, f"Last updated: {update_time}"
    except Exception as e:
        logger.error(f"Failed to update data: {e}")
        return {'updated': False}, "Update failed"


@app.callback(
    Output('kri-cards', 'children'),
    Input('data-store', 'data')
)
def update_kri_cards(data):
    """Update KRI summary cards."""
    if data_cache['kris'] is None:
        fetch_and_process_data()
    
    kris = data_cache['kris']
    risk_levels = data_cache['risk_levels']
    
    # Group by category
    credit_kris = ['loan_default_rate', 'delinquency_rate', 'credit_quality_score', 'loan_concentration']
    market_kris = ['portfolio_volatility', 'var_95', 'interest_rate_risk']
    liquidity_kris = ['liquidity_coverage_ratio', 'deposit_flow_ratio']
    
    def create_card(title, kri_names, icon):
        critical_count = sum(1 for k in kri_names if k in risk_levels and risk_levels[k].value == 'critical')
        high_count = sum(1 for k in kri_names if k in risk_levels and risk_levels[k].value == 'high')
        
        if critical_count > 0:
            color = RISK_COLORS['critical']
            status = 'CRITICAL'
        elif high_count > 0:
            color = RISK_COLORS['high']
            status = 'HIGH'
        else:
            color = RISK_COLORS['low']
            status = 'NORMAL'
        
        return html.Div([
            html.Div([
                html.Div(icon, style={'fontSize': '40px', 'marginBottom': '10px'}),
                html.H3(title, style={'margin': '0', 'fontSize': '18px', 'color': COLORS['text']}),
                html.Div(status, style={
                    'fontSize': '24px',
                    'fontWeight': 'bold',
                    'color': color,
                    'marginTop': '10px'
                }),
                html.Div(f"{len(kri_names)} indicators", style={
                    'fontSize': '14px',
                    'color': '#6c757d',
                    'marginTop': '5px'
                })
            ])
        ], style={
            'flex': '1',
            'backgroundColor': COLORS['card'],
            'padding': '25px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'margin': '0 10px',
            'border': f'3px solid {color}'
        })
    
    return html.Div([
        create_card('Credit Risk', credit_kris, '💳'),
        create_card('Market Risk', market_kris, '📈'),
        create_card('Liquidity Risk', liquidity_kris, '💧')
    ], style={'display': 'flex', 'gap': '20px'})


@app.callback(
    Output('economic-indicators-chart', 'figure'),
    Input('data-store', 'data')
)
def update_economic_chart(data):
    """Update economic indicators chart."""
    if data_cache['economic_data'] is None:
        fetch_and_process_data()
    
    df = data_cache['economic_data']
    
    fig = go.Figure()
    
    # Add traces for each indicator
    fig.add_trace(go.Scatter(
        x=df.index, y=df['unemployment'],
        name='Unemployment (%)',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['inflation'] * 100,
        name='Inflation (%)',
        line=dict(color=COLORS['danger'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['interest_rate'],
        name='Interest Rate (%)',
        line=dict(color=COLORS['success'], width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['credit_spread'],
        name='Credit Spread (%)',
        line=dict(color=COLORS['warning'], width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value (%)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    return fig


@app.callback(
    Output('forecasts-chart', 'figure'),
    Input('data-store', 'data')
)
def update_forecasts_chart(data):
    """Update forecasts chart with annotations."""
    if data_cache['economic_data'] is None or data_cache['forecasts'] is None:
        fetch_and_process_data()
    
    historical = data_cache['economic_data'].tail(24)
    forecasts = data_cache['forecasts']
    
    fig = go.Figure()
    
    colors = [COLORS['primary'], COLORS['danger'], COLORS['success'], COLORS['warning']]
    names = ['Unemployment', 'Inflation', 'Interest Rate', 'Credit Spread']
    
    for idx, col in enumerate(historical.columns):
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical[col] if col != 'inflation' else historical[col] * 100,
            name=f'{names[idx]} (Historical)',
            line=dict(color=colors[idx], width=2),
            showlegend=True,
            hovertemplate=f'{names[idx]}: %{{y:.2f}}%<extra></extra>'
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecasts.index,
            y=forecasts[col] if col != 'inflation' else forecasts[col] * 100,
            name=f'{names[idx]} (12-mo Forecast)',
            line=dict(color=colors[idx], width=2, dash='dash'),
            showlegend=True,
            hovertemplate=f'{names[idx]} Forecast: %{{y:.2f}}%<extra></extra>'
        ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=historical.index[-1],
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Economic Indicators: Historical Data + 12-Month Ensemble Forecasts<br><sub>Forecasts combine ARIMA, ETS, and LSTM models</sub>",
        xaxis_title="Date",
        yaxis_title="Value (%)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    return fig


@app.callback(
    Output('risk-heatmap', 'figure'),
    Input('data-store', 'data')
)
def update_risk_heatmap(data):
    """Update risk heatmap."""
    if data_cache['kris'] is None:
        fetch_and_process_data()
    
    kris = data_cache['kris']
    risk_levels = data_cache['risk_levels']
    
    # Create matrix for heatmap
    kri_names = list(kris.keys())
    risk_values = [risk_levels[k].value for k in kri_names]
    
    # Map risk levels to numbers
    risk_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
    risk_numbers = [risk_map[r] for r in risk_values]
    
    # Reshape for heatmap
    n_cols = 3
    n_rows = (len(kri_names) + n_cols - 1) // n_cols
    
    matrix = np.zeros((n_rows, n_cols))
    labels = [['' for _ in range(n_cols)] for _ in range(n_rows)]
    
    for idx, (name, value) in enumerate(zip(kri_names, risk_numbers)):
        row = idx // n_cols
        col = idx % n_cols
        matrix[row][col] = value
        labels[row][col] = name.replace('_', ' ').title()
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        text=labels,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorscale=[[0, 'white'], [0.25, RISK_COLORS['low']], 
                    [0.5, RISK_COLORS['medium']], [0.75, RISK_COLORS['high']], 
                    [1, RISK_COLORS['critical']]],
        showscale=False
    ))
    
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


@app.callback(
    Output('risk-distribution', 'figure'),
    Input('data-store', 'data')
)
def update_risk_distribution(data):
    """Update risk distribution pie chart."""
    if data_cache['risk_levels'] is None:
        fetch_and_process_data()
    
    risk_levels = data_cache['risk_levels']
    
    # Count by risk level
    counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
    for level in risk_levels.values():
        counts[level.value.title()] += 1
    
    fig = go.Figure(data=[go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        marker=dict(colors=[RISK_COLORS['low'], RISK_COLORS['medium'], 
                           RISK_COLORS['high'], RISK_COLORS['critical']]),
        hole=0.4
    )])
    
    fig.update_layout(
        annotations=[dict(text='Risk<br>Levels', x=0.5, y=0.5, font_size=16, showarrow=False)],
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
    )
    
    return fig


@app.callback(
    Output('model-comparison-chart', 'figure'),
    [Input('data-store', 'data'),
     Input('model-comparison-indicator', 'value')]
)
def update_model_comparison(data, indicator):
    """Update model comparison chart."""
    if data_cache['model_forecasts'] is None:
        fetch_and_process_data()
    
    historical = data_cache['economic_data'].tail(24)
    model_forecasts = data_cache['model_forecasts']
    
    if indicator not in model_forecasts:
        return go.Figure()
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical[indicator] if indicator != 'inflation' else historical[indicator] * 100,
        name='Historical',
        line=dict(color=COLORS['text'], width=3),
        mode='lines'
    ))
    
    # Model forecasts
    forecast_dates = data_cache['forecasts'].index
    colors_models = {'ARIMA': COLORS['primary'], 'ETS': COLORS['secondary'], 'Ensemble': COLORS['success']}
    
    for model_name, color in colors_models.items():
        if model_name in model_forecasts[indicator]:
            forecast_values = model_forecasts[indicator][model_name]
            if indicator == 'inflation':
                forecast_values = forecast_values * 100
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                name=model_name,
                line=dict(color=color, width=2, dash='dash' if model_name != 'Ensemble' else 'solid'),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title=f"{indicator.replace('_', ' ').title()} - Model Comparison",
        xaxis_title="Date",
        yaxis_title="Value (%)" if indicator != 'credit_spread' else "Spread (%)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    return fig


@app.callback(
    Output('scenario-chart', 'figure'),
    [Input('data-store', 'data'),
     Input('scenario-selector', 'value')]
)
def update_scenario_chart(data, scenario_name):
    """Update scenario analysis chart with detailed annotations."""
    if data_cache['scenario_results'] is None:
        fetch_and_process_data()
    
    scenario_results = data_cache['scenario_results']
    
    if scenario_name not in scenario_results or scenario_results[scenario_name] is None:
        return go.Figure()
    
    scenario_data = scenario_results[scenario_name]
    simulation_results = scenario_data['simulation']
    
    # Get scenario description
    scenario_descriptions = {
        'baseline': 'Normal economic conditions with small random fluctuations',
        'recession': 'Severe downturn: unemployment rises to 10%, GDP contracts',
        'rate_shock': 'Sudden interest rate increase from 3% to 6%',
        'credit_crisis': 'Credit market disruption: spreads widen from 2% to 8%'
    }
    
    fig = go.Figure()
    
    # Plot key simulation metrics over time
    if 'system_liquidity' in simulation_results.columns:
        fig.add_trace(go.Scatter(
            x=simulation_results.index,
            y=simulation_results['system_liquidity'],
            name='System Liquidity Ratio',
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate='Liquidity: %{y:.3f}<extra></extra>'
        ))
    
    if 'default_rate' in simulation_results.columns:
        fig.add_trace(go.Scatter(
            x=simulation_results.index,
            y=simulation_results['default_rate'] * 100,
            name='Firm Default Rate (%)',
            line=dict(color=COLORS['danger'], width=2),
            yaxis='y2',
            hovertemplate='Defaults: %{y:.1f}%<extra></extra>'
        ))
    
    if 'network_stress' in simulation_results.columns:
        fig.add_trace(go.Scatter(
            x=simulation_results.index,
            y=simulation_results['network_stress'],
            name='Network Stress Index',
            line=dict(color=COLORS['warning'], width=2),
            hovertemplate='Stress: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Agent-Based Simulation: {scenario_name.replace('_', ' ').title()}<br><sub>{scenario_descriptions.get(scenario_name, '')}</sub>",
        xaxis_title="Simulation Step (Monthly)",
        yaxis_title="Liquidity Ratio / Stress Index (0-1)",
        yaxis2=dict(
            title="Default Rate (%)",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    return fig


@app.callback(
    Output('kri-table', 'children'),
    Input('data-store', 'data')
)
def update_kri_table(data):
    """Update detailed KRI table with drill-down capability."""
    if data_cache['kris'] is None:
        fetch_and_process_data()
    
    kris = data_cache['kris']
    risk_levels = data_cache['risk_levels']
    
    rows = []
    for kri_name, value in kris.items():
        kri_def = kri_registry.get_kri(kri_name)
        risk_level = risk_levels[kri_name]
        
        # Get threshold info for drill-down
        thresholds = kri_def.thresholds
        threshold_text = f"Low: {thresholds['low']:.2f}, Med: {thresholds['medium']:.2f}, High: {thresholds['high']:.2f}, Crit: {thresholds['critical']:.2f}"
        
        row = html.Tr([
            html.Td(
                html.Div([
                    html.Div(kri_name.replace('_', ' ').title(), style={'fontWeight': 'bold'}),
                    html.Div(kri_def.description, style={'fontSize': '11px', 'color': '#6c757d', 'marginTop': '3px'}),
                    html.Div(f"Thresholds: {threshold_text}", style={'fontSize': '10px', 'color': '#6c757d', 'marginTop': '2px'})
                ]),
                style={'padding': '12px', 'borderBottom': '1px solid #dee2e6'}
            ),
            html.Td(kri_def.category.value.title(), style={'padding': '12px', 'borderBottom': '1px solid #dee2e6'}),
            html.Td(f"{value:.2f} {kri_def.unit}", style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'right', 'fontWeight': 'bold'}),
            html.Td(
                html.Span(risk_level.value.upper(), style={
                    'padding': '5px 10px',
                    'borderRadius': '4px',
                    'backgroundColor': RISK_COLORS[risk_level.value],
                    'color': 'white',
                    'fontWeight': 'bold',
                    'fontSize': '12px'
                }),
                style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'center'}
            ),
            html.Td(
                html.Span(
                    '📊 Leading' if kri_def.is_leading else '📈 Lagging',
                    style={'fontSize': '13px'}
                ),
                style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'center'}
            )
        ], style={'cursor': 'pointer', ':hover': {'backgroundColor': '#f8f9fa'}})
        rows.append(row)
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th('KRI Details', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'left', 'width': '40%'}),
            html.Th('Category', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'left'}),
            html.Th('Value', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'right'}),
            html.Th('Risk Level', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'center'}),
            html.Th('Type', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'center'})
        ])),
        html.Tbody(rows)
    ], style={'width': '100%', 'borderCollapse': 'collapse'})


@app.callback(
    Output('scenario-comparison-kris', 'figure'),
    Input('data-store', 'data')
)
def update_scenario_comparison_kris(data):
    """Update scenario comparison chart for KRIs."""
    if data_cache['scenario_results'] is None:
        fetch_and_process_data()
    
    scenario_results = data_cache['scenario_results']
    
    # Select key KRIs to compare
    key_kris = ['loan_default_rate', 'portfolio_volatility', 'liquidity_coverage_ratio']
    
    fig = go.Figure()
    
    scenarios = ['baseline', 'recession', 'rate_shock', 'credit_crisis']
    scenario_labels = ['Baseline', 'Recession', 'Rate Shock', 'Credit Crisis']
    
    for kri_name in key_kris:
        values = []
        for scenario_name in scenarios:
            if scenario_name in scenario_results and scenario_results[scenario_name] is not None:
                kri_value = scenario_results[scenario_name]['kris'].get(kri_name, 0)
                values.append(kri_value)
            else:
                values.append(0)
        
        fig.add_trace(go.Bar(
            name=kri_name.replace('_', ' ').title(),
            x=scenario_labels,
            y=values,
            text=[f"{v:.2f}" for v in values],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Key Risk Indicators Across Scenarios",
        xaxis_title="Scenario",
        yaxis_title="KRI Value",
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    return fig


@app.callback(
    Output('scenario-comparison-distribution', 'figure'),
    Input('data-store', 'data')
)
def update_scenario_distribution(data):
    """Update probability distribution across scenarios."""
    if data_cache['scenario_results'] is None:
        fetch_and_process_data()
    
    scenario_results = data_cache['scenario_results']
    
    # Calculate average default rate for each scenario
    scenarios = ['baseline', 'recession', 'rate_shock', 'credit_crisis']
    scenario_labels = ['Baseline', 'Recession', 'Rate Shock', 'Credit Crisis']
    default_rates = []
    
    for scenario_name in scenarios:
        if scenario_name in scenario_results and scenario_results[scenario_name] is not None:
            sim_results = scenario_results[scenario_name]['simulation']
            if 'default_rate' in sim_results.columns:
                avg_default = sim_results['default_rate'].mean() * 100
                default_rates.append(avg_default)
            else:
                default_rates.append(0)
        else:
            default_rates.append(0)
    
    # Create box plot showing distribution
    fig = go.Figure()
    
    colors_scenario = [COLORS['success'], COLORS['danger'], COLORS['warning'], COLORS['primary']]
    
    for idx, (scenario_name, label) in enumerate(zip(scenarios, scenario_labels)):
        if scenario_name in scenario_results and scenario_results[scenario_name] is not None:
            sim_results = scenario_results[scenario_name]['simulation']
            if 'default_rate' in sim_results.columns:
                fig.add_trace(go.Box(
                    y=sim_results['default_rate'] * 100,
                    name=label,
                    marker_color=colors_scenario[idx],
                    boxmean='sd'
                ))
    
    fig.update_layout(
        title="Default Rate Distribution by Scenario",
        yaxis_title="Default Rate (%)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
    
    return fig


@app.callback(
    Output('scenario-risk-metrics', 'children'),
    Input('data-store', 'data')
)
def update_scenario_risk_metrics(data):
    """Update scenario-conditional risk metrics table."""
    if data_cache['scenario_results'] is None:
        fetch_and_process_data()
    
    scenario_results = data_cache['scenario_results']
    
    scenarios = ['baseline', 'recession', 'rate_shock', 'credit_crisis']
    scenario_labels = ['Baseline', 'Recession', 'Rate Shock', 'Credit Crisis']
    
    rows = []
    
    for scenario_name, label in zip(scenarios, scenario_labels):
        if scenario_name not in scenario_results or scenario_results[scenario_name] is None:
            continue
        
        sim_results = scenario_results[scenario_name]['simulation']
        kris = scenario_results[scenario_name]['kris']
        
        # Calculate stress metrics
        if 'default_rate' in sim_results.columns:
            default_rates = sim_results['default_rate'] * 100
            mean_default = default_rates.mean()
            var_95 = np.percentile(default_rates, 95)  # Stress VaR
            tail_risk = default_rates[default_rates > var_95].mean()  # Conditional tail expectation
        else:
            mean_default = 0
            var_95 = 0
            tail_risk = 0
        
        # Get key KRIs
        loan_default = kris.get('loan_default_rate', 0)
        portfolio_vol = kris.get('portfolio_volatility', 0)
        lcr = kris.get('liquidity_coverage_ratio', 0)
        
        row = html.Tr([
            html.Td(label, style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'fontWeight': 'bold'}),
            html.Td(f"{mean_default:.2f}%", style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'right'}),
            html.Td(f"{var_95:.2f}%", style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'right'}),
            html.Td(f"{tail_risk:.2f}%", style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'right'}),
            html.Td(f"{loan_default:.2f}%", style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'right'}),
            html.Td(f"{portfolio_vol:.2f}%", style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'right'}),
            html.Td(f"{lcr:.2f}", style={'padding': '12px', 'borderBottom': '1px solid #dee2e6', 'textAlign': 'right'})
        ])
        rows.append(row)
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th('Scenario', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'left'}),
            html.Th('Mean Default Rate', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'right'}),
            html.Th('Stress VaR (95%)', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'right'}),
            html.Th('Tail Risk (CVaR)', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'right'}),
            html.Th('Loan Default Rate', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'right'}),
            html.Th('Portfolio Volatility', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'right'}),
            html.Th('Liquidity Ratio', style={'padding': '12px', 'borderBottom': '2px solid #dee2e6', 'textAlign': 'right'})
        ])),
        html.Tbody(rows)
    ], style={'width': '100%', 'borderCollapse': 'collapse'})


@app.callback(
    Output('download-data', 'data'),
    [Input('export-csv-button', 'n_clicks'),
     Input('export-excel-button', 'n_clicks'),
     Input('export-json-button', 'n_clicks')],
    prevent_initial_call=True
)
def export_data(csv_clicks, excel_clicks, json_clicks):
    """Export dashboard data in various formats."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if data_cache['kris'] is None:
        return None
    
    # Prepare export data
    kris = data_cache['kris']
    risk_levels = data_cache['risk_levels']
    economic_data = data_cache['economic_data']
    forecasts = data_cache['forecasts']
    
    # Create comprehensive export DataFrame
    export_data = {
        'KRI_Name': [],
        'Category': [],
        'Current_Value': [],
        'Risk_Level': [],
        'Type': []
    }
    
    for kri_name, value in kris.items():
        kri_def = kri_registry.get_kri(kri_name)
        risk_level = risk_levels[kri_name]
        
        export_data['KRI_Name'].append(kri_name)
        export_data['Category'].append(kri_def.category.value)
        export_data['Current_Value'].append(value)
        export_data['Risk_Level'].append(risk_level.value)
        export_data['Type'].append('Leading' if kri_def.is_leading else 'Lagging')
    
    df = pd.DataFrame(export_data)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if button_id == 'export-csv-button':
        return dcc.send_data_frame(df.to_csv, f"risk_dashboard_{timestamp}.csv", index=False)
    
    elif button_id == 'export-excel-button':
        # Create Excel with multiple sheets
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='KRIs', index=False)
            economic_data.to_excel(writer, sheet_name='Economic_Data')
            forecasts.to_excel(writer, sheet_name='Forecasts')
        
        output.seek(0)
        return dcc.send_bytes(output.getvalue(), f"risk_dashboard_{timestamp}.xlsx")
    
    elif button_id == 'export-json-button':
        # Create comprehensive JSON export
        json_data = {
            'timestamp': timestamp,
            'kris': {k: float(v) for k, v in kris.items()},
            'risk_levels': {k: v.value for k, v in risk_levels.items()},
            'economic_data': economic_data.tail(12).to_dict(orient='records'),
            'forecasts': forecasts.to_dict(orient='records')
        }
        
        import json
        json_str = json.dumps(json_data, indent=2)
        return dict(content=json_str, filename=f"risk_dashboard_{timestamp}.json")
    
    return None


if __name__ == '__main__':
    logger.info("Starting Risk Forecasting Dashboard...")
    logger.info(f"Dashboard will be available at: http://localhost:{settings.dashboard_port}")
    
    # Initial data load
    fetch_and_process_data()
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=settings.dashboard_port)
