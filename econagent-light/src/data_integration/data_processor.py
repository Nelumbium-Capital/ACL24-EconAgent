"""
Economic data processor for EconAgent-Light.
Processes FRED data and calibrates simulation parameters.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class EconomicDataProcessor:
    """
    Processes real economic data to calibrate simulation parameters.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.processed_data = {}
        self.calibration_params = {}
    
    def process_fred_data(self, fred_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process FRED data into simulation-ready format.
        
        Args:
            fred_data: Dictionary of FRED series DataFrames
            
        Returns:
            Dictionary of processed economic indicators
        """
        logger.info("Processing FRED data for simulation calibration")
        
        processed = {}
        
        # Process each series
        for name, df in fred_data.items():
            if df.empty:
                logger.warning(f"Empty data for {name}")
                continue
            
            try:
                # Basic processing
                series = df.iloc[:, 0]  # First column
                series = series.dropna()
                
                if len(series) == 0:
                    continue
                
                # Calculate statistics
                processed[name] = {
                    'data': series,
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'trend': self._calculate_trend(series),
                    'volatility': self._calculate_volatility(series),
                    'frequency': self._detect_frequency(series.index)
                }
                
                logger.debug(f"Processed {name}: {len(series)} observations")
                
            except Exception as e:
                logger.warning(f"Failed to process {name}: {e}")
                continue
        
        self.processed_data = processed
        return processed
    
    def calibrate_simulation_parameters(
        self,
        processed_data: Dict[str, Any],
        target_period: str = "2010-2023"
    ) -> Dict[str, Any]:
        """
        Calibrate simulation parameters based on real economic data.
        
        Args:
            processed_data: Processed FRED data
            target_period: Period to use for calibration
            
        Returns:
            Dictionary of calibrated simulation parameters
        """
        logger.info(f"Calibrating simulation parameters for period {target_period}")
        
        params = {
            # Economic parameters
            'productivity': 1.0,
            'max_price_inflation': 0.10,
            'max_wage_inflation': 0.05,
            'base_interest_rate': 0.02,
            
            # Agent parameters
            'n_agents': 100,
            'pareto_param': 8.0,
            'payment_max_skill_multiplier': 950.0,
            'labor_hours': 168,
            
            # Calibrated from real data
            'historical_stats': {}
        }
        
        # Calibrate inflation parameters
        if 'cpi' in processed_data:
            cpi_data = processed_data['cpi']['data']
            inflation_rate = cpi_data.pct_change(12).dropna()  # Year-over-year
            
            params['max_price_inflation'] = min(0.15, inflation_rate.std() * 3)
            params['target_inflation'] = 0.02  # Fed target
            params['historical_stats']['inflation'] = {
                'mean': inflation_rate.mean(),
                'std': inflation_rate.std(),
                'range': (inflation_rate.min(), inflation_rate.max())
            }
        
        # Calibrate unemployment parameters
        if 'unemployment' in processed_data:
            unemployment = processed_data['unemployment']['data'] / 100  # Convert to decimal
            
            params['natural_unemployment'] = unemployment.median()
            params['unemployment_volatility'] = unemployment.std()
            params['historical_stats']['unemployment'] = {
                'mean': unemployment.mean(),
                'std': unemployment.std(),
                'range': (unemployment.min(), unemployment.max())
            }
        
        # Calibrate interest rate parameters
        if 'fed_funds' in processed_data:
            fed_funds = processed_data['fed_funds']['data'] / 100  # Convert to decimal
            fed_funds = fed_funds.dropna()
            
            params['base_interest_rate'] = fed_funds.median()
            params['interest_rate_volatility'] = fed_funds.std()
            params['historical_stats']['interest_rate'] = {
                'mean': fed_funds.mean(),
                'std': fed_funds.std(),
                'range': (fed_funds.min(), fed_funds.max())
            }
        
        # Calibrate GDP growth parameters
        if 'real_gdp' in processed_data:
            gdp = processed_data['real_gdp']['data']
            gdp_growth = gdp.pct_change(4).dropna()  # Quarterly growth annualized
            
            params['productivity'] = max(0.5, 1.0 + gdp_growth.mean())
            params['gdp_volatility'] = gdp_growth.std()
            params['historical_stats']['gdp_growth'] = {
                'mean': gdp_growth.mean(),
                'std': gdp_growth.std(),
                'range': (gdp_growth.min(), gdp_growth.max())
            }
        
        # Calibrate wage parameters
        if 'wages' in processed_data:
            wages = processed_data['wages']['data']
            wage_growth = wages.pct_change(12).dropna()  # Year-over-year
            
            params['max_wage_inflation'] = min(0.10, wage_growth.std() * 2)
            params['historical_stats']['wage_growth'] = {
                'mean': wage_growth.mean(),
                'std': wage_growth.std(),
                'range': (wage_growth.min(), wage_growth.max())
            }
        
        # Calculate Phillips curve relationship
        if 'unemployment' in processed_data and 'cpi' in processed_data:
            phillips_coeff = self._estimate_phillips_curve(
                processed_data['unemployment']['data'],
                processed_data['cpi']['data']
            )
            params['phillips_curve_coefficient'] = phillips_coeff
        
        # Calculate Okun's law relationship
        if 'unemployment' in processed_data and 'real_gdp' in processed_data:
            okun_coeff = self._estimate_okuns_law(
                processed_data['unemployment']['data'],
                processed_data['real_gdp']['data']
            )
            params['okuns_law_coefficient'] = okun_coeff
        
        self.calibration_params = params
        logger.info("Simulation parameters calibrated successfully")
        
        return params
    
    def create_validation_targets(
        self,
        processed_data: Dict[str, Any],
        validation_period: str = "2020-2023"
    ) -> Dict[str, Any]:
        """
        Create validation targets from real data for model validation.
        
        Args:
            processed_data: Processed FRED data
            validation_period: Period to use for validation
            
        Returns:
            Dictionary of validation targets
        """
        logger.info(f"Creating validation targets for period {validation_period}")
        
        targets = {}
        
        # Extract validation period data
        start_date = pd.to_datetime(validation_period.split('-')[0])
        end_date = pd.to_datetime(validation_period.split('-')[1])
        
        for name, data_dict in processed_data.items():
            series = data_dict['data']
            
            # Filter to validation period
            mask = (series.index >= start_date) & (series.index <= end_date)
            validation_series = series[mask]
            
            if len(validation_series) == 0:
                continue
            
            targets[name] = {
                'mean': validation_series.mean(),
                'std': validation_series.std(),
                'trend': self._calculate_trend(validation_series),
                'correlation_targets': {}
            }
        
        # Calculate correlation targets
        for name1 in targets:
            for name2 in targets:
                if name1 != name2:
                    series1 = processed_data[name1]['data']
                    series2 = processed_data[name2]['data']
                    
                    # Align series
                    aligned = pd.concat([series1, series2], axis=1).dropna()
                    if len(aligned) > 10:
                        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                        targets[name1]['correlation_targets'][name2] = corr
        
        return targets
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend of a time series."""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 0.0
        
        slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
        return slope
    
    def _calculate_volatility(self, series: pd.Series, window: int = 12) -> float:
        """Calculate rolling volatility of a time series."""
        if len(series) < window:
            return series.std()
        
        returns = series.pct_change().dropna()
        rolling_vol = returns.rolling(window=window).std()
        return rolling_vol.mean()
    
    def _detect_frequency(self, index: pd.DatetimeIndex) -> str:
        """Detect the frequency of a time series."""
        if len(index) < 2:
            return 'unknown'
        
        diff = index[1] - index[0]
        
        if diff.days <= 1:
            return 'daily'
        elif diff.days <= 7:
            return 'weekly'
        elif diff.days <= 31:
            return 'monthly'
        elif diff.days <= 93:
            return 'quarterly'
        else:
            return 'annual'
    
    def _estimate_phillips_curve(
        self,
        unemployment: pd.Series,
        cpi: pd.Series
    ) -> float:
        """Estimate Phillips curve coefficient."""
        try:
            # Calculate inflation rate
            inflation = cpi.pct_change(12).dropna() * 100  # Year-over-year %
            
            # Align series
            aligned = pd.concat([unemployment, inflation], axis=1).dropna()
            
            if len(aligned) < 10:
                return -0.5  # Default Phillips curve coefficient
            
            # Regression: inflation = a + b * unemployment
            X = aligned.iloc[:, 0].values.reshape(-1, 1)  # unemployment
            y = aligned.iloc[:, 1].values  # inflation
            
            reg = LinearRegression().fit(X, y)
            coefficient = reg.coef_[0]
            
            logger.info(f"Phillips curve coefficient: {coefficient:.3f}")
            return coefficient
            
        except Exception as e:
            logger.warning(f"Failed to estimate Phillips curve: {e}")
            return -0.5
    
    def _estimate_okuns_law(
        self,
        unemployment: pd.Series,
        gdp: pd.Series
    ) -> float:
        """Estimate Okun's law coefficient."""
        try:
            # Calculate GDP growth and unemployment change
            gdp_growth = gdp.pct_change(4).dropna() * 100  # Quarterly growth %
            unemployment_change = unemployment.diff().dropna()
            
            # Align series
            aligned = pd.concat([gdp_growth, unemployment_change], axis=1).dropna()
            
            if len(aligned) < 10:
                return -0.3  # Default Okun's coefficient
            
            # Regression: unemployment_change = a + b * gdp_growth
            X = aligned.iloc[:, 0].values.reshape(-1, 1)  # gdp_growth
            y = aligned.iloc[:, 1].values  # unemployment_change
            
            reg = LinearRegression().fit(X, y)
            coefficient = reg.coef_[0]
            
            logger.info(f"Okun's law coefficient: {coefficient:.3f}")
            return coefficient
            
        except Exception as e:
            logger.warning(f"Failed to estimate Okun's law: {e}")
            return -0.3
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of processed data and calibration."""
        if not self.processed_data or not self.calibration_params:
            return "No data processed yet."
        
        report = []
        report.append("=== FRED Data Processing Summary ===\n")
        
        # Data summary
        report.append(f"Processed {len(self.processed_data)} economic series:")
        for name, data in self.processed_data.items():
            report.append(f"  • {name}: {len(data['data'])} observations")
            report.append(f"    Mean: {data['mean']:.3f}, Std: {data['std']:.3f}")
            report.append(f"    Trend: {data['trend']:.6f}, Volatility: {data['volatility']:.3f}")
        
        report.append("\n=== Calibrated Parameters ===")
        
        # Key parameters
        key_params = [
            'productivity', 'max_price_inflation', 'max_wage_inflation',
            'base_interest_rate', 'natural_unemployment', 'phillips_curve_coefficient',
            'okuns_law_coefficient'
        ]
        
        for param in key_params:
            if param in self.calibration_params:
                value = self.calibration_params[param]
                report.append(f"  • {param}: {value:.4f}")
        
        # Historical statistics
        if 'historical_stats' in self.calibration_params:
            report.append("\n=== Historical Statistics ===")
            for indicator, stats in self.calibration_params['historical_stats'].items():
                report.append(f"  • {indicator}:")
                report.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                report.append(f"    Range: [{stats['range'][0]:.4f}, {stats['range'][1]:.4f}]")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    from fred_client import FREDClient
    
    # Initialize clients
    fred = FREDClient()
    processor = EconomicDataProcessor()
    
    try:
        # Fetch sample data
        fred_data = fred.get_core_economic_data(start_date='2015-01-01')
        
        # Process data
        processed = processor.process_fred_data(fred_data)
        
        # Calibrate parameters
        params = processor.calibrate_simulation_parameters(processed)
        
        # Generate report
        report = processor.generate_summary_report()
        print(report)
        
    except Exception as e:
        print(f"Error: {e}")