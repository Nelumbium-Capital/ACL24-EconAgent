"""
Real Data Manager for EconAgent-Light.
Integrates FRED data into the simulation with no mock data or placeholders.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path

from .fred_client import FREDClient
from .data_processor import EconomicDataProcessor
from .calibration import ModelCalibrator

logger = logging.getLogger(__name__)

class RealDataManager:
    """
    Manages real economic data integration for EconAgent-Light simulation.
    Replaces all mock data with actual FRED data.
    """
    
    def __init__(
        self,
        fred_api_key: str,
        cache_dir: str = "./data_cache",
        auto_update: bool = True
    ):
        """
        Initialize real data manager with FRED integration.
        
        Args:
            fred_api_key: FRED API key for data access
            cache_dir: Directory for caching FRED data
            auto_update: Whether to automatically update data
        """
        self.fred_client = FREDClient(
            api_key=fred_api_key,
            cache_dir=cache_dir
        )
        self.data_processor = EconomicDataProcessor()
        self.calibrator = ModelCalibrator(fred_api_key, cache_dir)
        
        self.auto_update = auto_update
        self.last_update = None
        
        # Real economic data storage
        self.current_data = {}
        self.historical_data = {}
        self.calibrated_params = {}
        
        # Data update tracking
        self.data_freshness = {}
        
        logger.info("RealDataManager initialized with FRED integration")
    
    def initialize_real_data(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        calibration_scenario: str = "post_covid"
    ) -> Dict[str, Any]:
        """
        Initialize simulation with real economic data.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for data (defaults to today)
            calibration_scenario: Scenario for parameter calibration
            
        Returns:
            Dictionary of real economic parameters and data
        """
        logger.info("Initializing simulation with real FRED data")
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Step 1: Fetch comprehensive FRED data
        self.historical_data = self.fred_client.get_core_economic_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if not self.historical_data:
            raise ValueError("Failed to fetch FRED data. Check API key and connection.")
        
        # Step 2: Process and calibrate using real data
        self.calibrated_params = self.calibrator.calibrate_for_scenario(
            scenario=calibration_scenario,
            custom_period=(start_date, end_date)
        )
        
        # Step 3: Extract current economic conditions
        self.current_data = self._extract_current_conditions()
        
        # Step 4: Create real-time economic context
        economic_context = self._create_economic_context()
        
        self.last_update = datetime.now()
        
        logger.info(f"Real data initialization complete. Using {len(self.historical_data)} FRED series.")
        
        return {
            'calibrated_params': self.calibrated_params,
            'current_conditions': self.current_data,
            'economic_context': economic_context,
            'data_sources': list(self.historical_data.keys()),
            'last_update': self.last_update.isoformat()
        }
    
    def get_real_time_indicators(self) -> Dict[str, float]:
        """
        Get current real-time economic indicators from FRED.
        
        Returns:
            Dictionary of current economic indicators
        """
        # Update data if needed
        if self.auto_update and self._needs_update():
            self._update_current_data()
        
        indicators = {}
        
        # Extract latest values from each series
        for name, df in self.historical_data.items():
            if not df.empty:
                latest_value = df.iloc[-1, 0]  # Most recent value
                latest_date = df.index[-1]
                
                indicators[name] = {
                    'value': float(latest_value),
                    'date': latest_date.strftime("%Y-%m-%d"),
                    'series_id': self.fred_client.CORE_SERIES.get(name, 'Unknown')
                }
        
        # Calculate derived indicators
        indicators.update(self._calculate_derived_indicators())
        
        return indicators
    
    def get_economic_forecast(
        self,
        months_ahead: int = 12,
        confidence_level: float = 0.95
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate economic forecasts based on historical FRED data.
        
        Args:
            months_ahead: Number of months to forecast
            confidence_level: Confidence level for forecast intervals
            
        Returns:
            Dictionary of forecasted economic indicators
        """
        forecasts = {}
        
        for name, df in self.historical_data.items():
            if len(df) < 24:  # Need at least 2 years of data
                continue
            
            try:
                forecast = self._forecast_series(
                    df.iloc[:, 0], 
                    months_ahead, 
                    confidence_level
                )
                forecasts[name] = forecast
                
            except Exception as e:
                logger.warning(f"Failed to forecast {name}: {e}")
                continue
        
        return forecasts
    
    def update_simulation_parameters(
        self,
        current_step: int,
        simulation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update simulation parameters based on real economic conditions.
        
        Args:
            current_step: Current simulation step
            simulation_data: Current simulation state
            
        Returns:
            Updated parameters based on real data
        """
        # Get current real indicators
        real_indicators = self.get_real_time_indicators()
        
        # Calculate parameter adjustments
        adjustments = {}
        
        # Adjust interest rates based on real Fed funds rate
        if 'fed_funds' in real_indicators:
            real_rate = real_indicators['fed_funds']['value'] / 100
            adjustments['interest_rate'] = real_rate
        
        # Adjust inflation expectations based on real CPI
        if 'cpi' in real_indicators:
            # Calculate recent inflation trend
            cpi_data = self.historical_data['cpi'].iloc[:, 0]
            recent_inflation = cpi_data.pct_change(12).iloc[-1]  # Year-over-year
            adjustments['inflation_expectation'] = recent_inflation
        
        # Adjust unemployment based on real data
        if 'unemployment' in real_indicators:
            real_unemployment = real_indicators['unemployment']['value'] / 100
            adjustments['unemployment_rate'] = real_unemployment
        
        # Adjust productivity based on real GDP growth
        if 'real_gdp' in real_indicators:
            gdp_data = self.historical_data['real_gdp'].iloc[:, 0]
            recent_growth = gdp_data.pct_change(4).iloc[-1]  # Quarterly growth
            adjustments['productivity_growth'] = recent_growth
        
        return adjustments
    
    def validate_simulation_results(
        self,
        simulation_results: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Validate simulation results against real FRED data.
        
        Args:
            simulation_results: DataFrame of simulation results
            
        Returns:
            Dictionary of validation metrics
        """
        return self.calibrator.validate_calibration(simulation_results)
    
    def generate_data_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive report on real data integration.
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 70)
        report.append("ECONAGENT-LIGHT REAL DATA INTEGRATION REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Last Data Update: {self.last_update}")
        report.append("")
        
        # FRED Data Summary
        report.append("FRED DATA SOURCES:")
        report.append("-" * 25)
        for name, df in self.historical_data.items():
            series_id = self.fred_client.CORE_SERIES.get(name, 'Unknown')
            start_date = df.index[0].strftime("%Y-%m-%d") if not df.empty else "N/A"
            end_date = df.index[-1].strftime("%Y-%m-%d") if not df.empty else "N/A"
            report.append(f"• {name.upper():<20}: {series_id} ({start_date} to {end_date})")
        report.append("")
        
        # Current Economic Conditions
        if self.current_data:
            report.append("CURRENT ECONOMIC CONDITIONS:")
            report.append("-" * 35)
            for indicator, data in self.current_data.items():
                if isinstance(data, dict) and 'value' in data:
                    value = data['value']
                    date = data.get('date', 'Unknown')
                    report.append(f"• {indicator.replace('_', ' ').title():<25}: {value:>8.3f} (as of {date})")
        report.append("")
        
        # Calibrated Parameters
        if self.calibrated_params:
            report.append("CALIBRATED PARAMETERS (FROM REAL DATA):")
            report.append("-" * 45)
            
            key_params = [
                ('Base Interest Rate', 'base_interest_rate'),
                ('Natural Unemployment', 'natural_unemployment'),
                ('Max Price Inflation', 'max_price_inflation'),
                ('Max Wage Inflation', 'max_wage_inflation'),
                ('Productivity', 'productivity'),
                ('Phillips Curve Coeff', 'phillips_curve_coefficient'),
                ('Okun\'s Law Coeff', 'okuns_law_coefficient')
            ]
            
            for label, key in key_params:
                if key in self.calibrated_params:
                    value = self.calibrated_params[key]
                    report.append(f"• {label:<25}: {value:>8.4f}")
        
        report.append("")
        
        # Data Quality Assessment
        report.append("DATA QUALITY ASSESSMENT:")
        report.append("-" * 30)
        
        total_series = len(self.historical_data)
        complete_series = sum(1 for df in self.historical_data.values() if not df.empty)
        coverage = (complete_series / total_series * 100) if total_series > 0 else 0
        
        report.append(f"• Total FRED Series: {total_series}")
        report.append(f"• Complete Series: {complete_series}")
        report.append(f"• Data Coverage: {coverage:.1f}%")
        
        if self.last_update:
            hours_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
            report.append(f"• Hours Since Update: {hours_since_update:.1f}")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Data report saved to {output_path}")
        
        return report_text
    
    def _extract_current_conditions(self) -> Dict[str, Any]:
        """Extract current economic conditions from latest FRED data."""
        current = {}
        
        for name, df in self.historical_data.items():
            if not df.empty:
                latest_value = df.iloc[-1, 0]
                latest_date = df.index[-1]
                
                current[name] = {
                    'value': float(latest_value),
                    'date': latest_date.strftime("%Y-%m-%d"),
                    'change_1m': self._calculate_change(df, 1),
                    'change_3m': self._calculate_change(df, 3),
                    'change_12m': self._calculate_change(df, 12)
                }
        
        return current
    
    def _create_economic_context(self) -> Dict[str, Any]:
        """Create economic context for agent decision-making."""
        context = {
            'market_conditions': 'normal',
            'policy_stance': 'neutral',
            'economic_cycle': 'expansion',
            'risk_factors': []
        }
        
        # Analyze current conditions to determine context
        if 'unemployment' in self.current_data:
            unemployment = self.current_data['unemployment']['value']
            if unemployment > 7.0:
                context['market_conditions'] = 'recession'
                context['economic_cycle'] = 'contraction'
                context['risk_factors'].append('high_unemployment')
            elif unemployment < 4.0:
                context['market_conditions'] = 'tight_labor'
                context['risk_factors'].append('labor_shortage')
        
        if 'fed_funds' in self.current_data:
            fed_rate = self.current_data['fed_funds']['value']
            if fed_rate > 4.0:
                context['policy_stance'] = 'restrictive'
                context['risk_factors'].append('high_interest_rates')
            elif fed_rate < 1.0:
                context['policy_stance'] = 'accommodative'
        
        if 'cpi' in self.current_data:
            inflation_change = self.current_data['cpi']['change_12m']
            if inflation_change > 4.0:
                context['risk_factors'].append('high_inflation')
            elif inflation_change < 0:
                context['risk_factors'].append('deflation_risk')
        
        return context
    
    def _calculate_change(self, df: pd.DataFrame, months: int) -> float:
        """Calculate percentage change over specified months."""
        if len(df) < months + 1:
            return 0.0
        
        current = df.iloc[-1, 0]
        previous = df.iloc[-(months + 1), 0]
        
        if previous == 0:
            return 0.0
        
        return ((current - previous) / previous) * 100
    
    def _calculate_derived_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Calculate derived economic indicators."""
        derived = {}
        
        # Real interest rate (Fed funds - inflation)
        if 'fed_funds' in self.current_data and 'cpi' in self.current_data:
            nominal_rate = self.current_data['fed_funds']['value']
            inflation_rate = self.current_data['cpi']['change_12m']
            real_rate = nominal_rate - inflation_rate
            
            derived['real_interest_rate'] = {
                'value': real_rate,
                'date': self.current_data['fed_funds']['date'],
                'calculation': f"{nominal_rate:.2f}% - {inflation_rate:.2f}%"
            }
        
        # Yield curve slope (10Y - 3M)
        if 'treasury_10y' in self.current_data and 'treasury_3m' in self.current_data:
            ten_year = self.current_data['treasury_10y']['value']
            three_month = self.current_data['treasury_3m']['value']
            slope = ten_year - three_month
            
            derived['yield_curve_slope'] = {
                'value': slope,
                'date': self.current_data['treasury_10y']['date'],
                'interpretation': 'inverted' if slope < 0 else 'normal'
            }
        
        return derived
    
    def _forecast_series(
        self,
        series: pd.Series,
        months_ahead: int,
        confidence_level: float
    ) -> Dict[str, float]:
        """Simple forecast using exponential smoothing."""
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        
        # Calculate trend
        trend = series.diff().mean()
        
        # Last value
        last_value = series.iloc[-1]
        
        # Forecast
        forecast_value = last_value + (trend * months_ahead)
        
        # Confidence interval (simplified)
        std_error = series.std()
        margin = 1.96 * std_error  # 95% confidence
        
        return {
            'forecast': forecast_value,
            'lower_bound': forecast_value - margin,
            'upper_bound': forecast_value + margin,
            'trend': trend
        }
    
    def _needs_update(self) -> bool:
        """Check if data needs updating."""
        if not self.last_update:
            return True
        
        # Update daily
        hours_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
        return hours_since_update > 24
    
    def _update_current_data(self):
        """Update current data from FRED."""
        try:
            # Fetch latest data for key indicators
            key_series = ['UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'GDP']
            
            for series_id in key_series:
                latest_data = self.fred_client.get_series(
                    series_id,
                    start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                )
                
                # Update historical data
                series_name = None
                for name, sid in self.fred_client.CORE_SERIES.items():
                    if sid == series_id:
                        series_name = name
                        break
                
                if series_name and not latest_data.empty:
                    self.historical_data[series_name] = latest_data
            
            # Update current conditions
            self.current_data = self._extract_current_conditions()
            self.last_update = datetime.now()
            
            logger.info("Real-time data updated successfully")
            
        except Exception as e:
            logger.warning(f"Failed to update real-time data: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test with the provided API key
    api_key = "bcc1a43947af1745a35bfb3b7132b7c6"
    
    try:
        # Initialize real data manager
        data_manager = RealDataManager(fred_api_key=api_key)
        
        # Initialize with real data
        real_data = data_manager.initialize_real_data(
            start_date="2020-01-01",
            calibration_scenario="post_covid"
        )
        
        print("Real Data Integration Successful!")
        print(f"Loaded {len(real_data['data_sources'])} FRED series")
        
        # Get current indicators
        indicators = data_manager.get_real_time_indicators()
        print(f"\nCurrent Economic Indicators:")
        for name, data in list(indicators.items())[:5]:  # Show first 5
            if isinstance(data, dict) and 'value' in data:
                print(f"• {name}: {data['value']:.3f}")
        
        # Generate report
        report = data_manager.generate_data_report()
        print("\n" + "="*50)
        print("DATA INTEGRATION REPORT")
        print("="*50)
        print(report[:1000] + "..." if len(report) > 1000 else report)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Check your FRED API key and internet connection")