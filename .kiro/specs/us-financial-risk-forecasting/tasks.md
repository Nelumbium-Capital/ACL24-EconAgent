# Implementation Plan

- [x] 1. Set up project structure and core infrastructure
  - Create directory structure for data, models, simulation, agents, dashboard, and tests modules
  - Implement configuration management using Pydantic BaseSettings with .env file support
  - Set up logging infrastructure with rotating file handlers and console output
  - Create requirements.txt with all dependencies (pandas, numpy, statsmodels, torch, mesa, dash, fredapi, requests)
  - _Requirements: 8.1, 8.2, 11.1_

- [x] 2. Implement FRED data acquisition and caching system
- [x] 2.1 Create FRED API client with caching
  - Write FREDClient class with fetch_series() and fetch_multiple_series() methods
  - Implement local file-based caching with timestamp tracking to minimize API calls
  - Add retry logic with exponential backoff for network failures
  - Implement cache validation and staleness detection
  - _Requirements: 1.1, 1.4, 1.5_

- [x] 2.2 Build data pipeline for ETL operations
  - Create DataPipeline class for cleaning and aligning time-series data
  - Implement missing value handling (interpolation, forward-fill) with configurable strategies
  - Add frequency alignment to convert all series to consistent monthly/quarterly frequency
  - Implement data versioning with metadata tracking (fetch date, series IDs, date ranges)
  - _Requirements: 1.2, 1.3, 1.5_

- [ ]* 2.3 Write unit tests for data acquisition
  - Test FRED client with mocked API responses
  - Test cache hit/miss scenarios
  - Test data pipeline transformations
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 3. Define KRI framework and calculation engine
- [x] 3.1 Create KRI definition system
  - Define KRIDefinition dataclass with name, category, thresholds, and calculation method
  - Create KRI registry for credit risk indicators (default rate, delinquency, credit quality, concentration)
  - Create KRI registry for market risk indicators (volatility, VaR, interest rate risk)
  - Create KRI registry for liquidity risk indicators (LCR, deposit flow ratio)
  - Document each KRI with data sources and threshold rationale
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3.2 Implement KRI calculator
  - Write KRICalculator class with compute_credit_kris(), compute_market_kris(), compute_liquidity_kris() methods
  - Implement threshold evaluation logic with risk level classification (low, medium, high, critical)
  - Add trend detection (improving, stable, deteriorating) based on historical values
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 4. Implement classical time-series forecasting models
- [x] 4.1 Create base forecaster interface
  - Define BaseForecaster abstract class with fit() and forecast() methods
  - Implement common utilities for data preparation and validation
  - _Requirements: 3.3_

- [x] 4.2 Implement ARIMA/SARIMA models
  - Create ARIMAForecaster class using statsmodels
  - Implement automatic order selection using AIC/BIC
  - Add seasonal decomposition for SARIMA
  - _Requirements: 3.3_

- [x] 4.3 Implement exponential smoothing models
  - Create ETSForecaster class for exponential smoothing
  - Support trend and seasonal components
  - _Requirements: 3.3_

- [ ]* 4.4 Write unit tests for classical models
  - Test model fitting and forecasting
  - Test forecast shape and validity
  - _Requirements: 3.3_

- [x] 5. Implement deep learning forecasting models
- [x] 5.1 Build Deep VAR model
  - Create DeepVARModel class with PyTorch neural network architecture
  - Implement multi-layer feedforward network with ReLU activations and dropout
  - Add training loop with validation split and early stopping
  - Implement recursive multi-step forecasting
  - _Requirements: 3.1, 3.2_

- [x] 5.2 Build LSTM forecaster
  - Create LSTMForecaster class with PyTorch LSTM layers
  - Implement sliding window sequence preparation
  - Add training with gradient clipping and learning rate scheduling
  - Implement recursive forecasting for multi-step predictions
  - _Requirements: 3.2_

- [x] 5.3 Implement ensemble forecaster
  - Create EnsembleForecaster class that combines multiple model predictions
  - Implement weighted averaging with configurable weights
  - Add weight optimization based on validation performance
  - Support dynamic weight adjustment based on recent accuracy
  - _Requirements: 3.4_

- [ ]* 5.4 Write unit tests for deep learning models
  - Test model initialization and architecture
  - Test training convergence
  - Test forecast output shapes
  - _Requirements: 3.1, 3.2, 3.4_

- [x] 6. Implement calibration and backtesting engine
- [x] 6.1 Create calibration engine
  - Write CalibrationEngine class with backtest() method using time-series cross-validation
  - Implement performance metrics calculation (MAE, RMSE, MAPE, directional accuracy)
  - Add hyperparameter optimization using grid search
  - Implement model comparison and selection logic
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 6.2 Implement automated retraining system
  - Create auto_retrain() method with configurable triggers (time-based, performance-based)
  - Add model versioning and rollback capability
  - Implement validation before deploying retrained models
  - Add logging for retraining events and performance tracking
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [ ]* 6.3 Write integration tests for calibration
  - Test end-to-end backtesting pipeline
  - Test hyperparameter optimization
  - Test automated retraining triggers
  - _Requirements: 6.1, 6.2, 10.1_

- [x] 7. Build Mesa-based stress testing simulation
- [x] 7.1 Create simulation model and agents
  - Implement RiskSimulationModel class extending mesa.Model with scheduler and data collector
  - Create BankAgent class with balance sheet, capital ratio, liquidity ratio, and lending decisions
  - Create FirmAgent class with borrowing needs and default probability
  - Implement agent interaction logic for loan origination and defaults
  - _Requirements: 4.1, 4.5_

- [x] 7.2 Implement scenario generator
  - Create EconomicScenario base class with apply_shock() method
  - Implement RecessionScenario with unemployment spike and GDP contraction
  - Implement InterestRateShockScenario with sudden rate increases
  - Implement CreditCrisisScenario with credit spread widening
  - Add custom scenario support with user-defined shock paths
  - _Requirements: 4.2, 12.1, 12.2_

- [x] 7.3 Add Monte Carlo simulation capability
  - Implement run_monte_carlo() method to run multiple simulation instances
  - Add random seed management for reproducibility
  - Compute probability distributions of KRI outcomes across simulations
  - Generate summary statistics (mean, median, percentiles) for stress scenarios
  - _Requirements: 4.3, 12.3_

- [ ]* 7.4 Write unit tests for simulation
  - Test agent initialization and behavior
  - Test scenario shock application
  - Test Monte Carlo aggregation
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 8. Implement event-driven agent architecture
- [x] 8.1 Create event bus and event schemas
  - Implement EventBus class with subscribe() and publish() methods
  - Define event dataclasses (DataUpdateEvent, ForecastCompleteEvent, SimulationCompleteEvent, KRIUpdateEvent, RiskSignalEvent)
  - Add event logging and debugging capabilities
  - _Requirements: 5.1, 5.4_

- [x] 8.2 Build Risk Manager agent
  - Create RiskManagerAgent class that subscribes to data and forecast events
  - Implement on_data_update(), on_forecast_complete(), on_simulation_complete() event handlers
  - Add KRI computation and threshold evaluation logic
  - Publish KRIUpdateEvent with alerts when thresholds are breached
  - _Requirements: 5.1, 5.3_

- [x] 8.3 Build Market Analyst agent
  - Create MarketAnalystAgent class that subscribes to data update events
  - Implement volatility spike detection using rolling standard deviation
  - Implement credit spread widening detection
  - Publish RiskSignalEvent when anomalies are detected
  - _Requirements: 5.2_

- [ ]* 8.4 Write integration tests for agent system
  - Test event publishing and subscription
  - Test agent communication flow
  - Test end-to-end event-driven pipeline
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 9. Build risk visualization dashboard
- [x] 9.1 Create dashboard layout and components
  - Implement RiskDashboard class using Dash framework
  - Create KRI summary cards showing current values and risk levels
  - Add time-series charts for KRI history with Plotly
  - Create forecast comparison charts showing multiple model predictions
  - Add scenario analysis dropdown and comparison visualizations
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 9.2 Implement interactive features
  - Add drill-down capabilities to view underlying data and model details
  - Implement auto-refresh with configurable intervals using dcc.Interval
  - Add export functionality for CSV, Excel, and JSON formats
  - Create alert highlighting for KRIs exceeding thresholds with color coding
  - _Requirements: 7.2, 7.3, 7.4_

- [x] 9.3 Add scenario comparison views
  - Create side-by-side scenario comparison charts
  - Display probability-weighted outcomes across scenarios
  - Show scenario-conditional risk metrics (stress VaR, tail risk)
  - _Requirements: 12.3, 12.4, 12.5_

- [ ]* 9.4 Write UI tests for dashboard
  - Test dashboard rendering
  - Test interactive callbacks
  - Test data export functionality
  - _Requirements: 7.1, 7.4_

- [ ] 10. Implement error handling and resilience
  - Add ErrorHandler class with handle_data_error() and handle_model_error() methods
  - Implement cache fallback for FRED API failures with staleness warnings
  - Add retry logic with exponential backoff for transient failures
  - Implement graceful degradation for model failures (fall back to simpler models)
  - Add comprehensive error logging with context information
  - _Requirements: 11.2, 11.4_

- [ ] 11. Create comprehensive documentation
  - Write README.md with system overview, setup instructions, and usage examples
  - Document all KRI definitions with calculation methods and data sources
  - Create API documentation for all public classes and methods
  - Write deployment guide with environment configuration and troubleshooting
  - Document model architectures, hyperparameters, and performance benchmarks
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 12. Plan WRDS integration architecture
  - Design WRDSClient interface with methods for CRSP and Compustat data
  - Document WRDS data requirements (equity prices, fundamentals, bond data)
  - Create modular data ingestion interfaces to accommodate future WRDS sources
  - Identify KRIs that will benefit from WRDS data (firm-level credit metrics)
  - Document WRDS access requirements and authentication flow
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 13. Implement end-to-end integration and validation
  - Create main orchestration script that runs full pipeline (data fetch → forecast → stress test → dashboard)
  - Implement integration test that validates complete workflow from data to visualization
  - Add performance benchmarks for forecast latency and simulation scalability
  - Create example notebooks demonstrating key use cases
  - Validate system runs entirely locally without external dependencies
  - _Requirements: 11.1, 11.3, 11.5_
