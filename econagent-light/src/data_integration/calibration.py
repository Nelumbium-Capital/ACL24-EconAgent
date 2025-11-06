"""
Model calibration module for EconAgent-Light.
Calibrates simulation parameters using real FRED data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .fred_client import FREDClient
from .data_processor import EconomicDataProcessor

logger = logging.getLogger(__name__)

class ModelCalibrator:
    """
    Calibrates EconAgent-Light model parameters using real economic data.
    """
    
    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_dir: str = "./data_cache"
    ):
        """
        Initialize model calibrator.
        
        Args:
            fred_api_key: FRED API key for data access
            cache_dir: Directory for caching data
        """
        self.fred_client = FREDClient(api_key=fred_api_key, cache_dir=cache_dir)
        self.data_processor = EconomicDataProcessor()
        
        self.raw_data = {}
        self.processed_data = {}
        self.calibrated_params = {}
        self.validation_targets = {}
    
    def calibrate_from_period(
        self,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Calibrate model parameters from a specific time period.
        
        Args:
            start_date: Start date for calibration data
            end_date: End date for calibration data (defaults to today)
            validation_split: Fraction of data to reserve for validation
            
        Returns:
            Dictionary of calibrated parameters
        """
        logger.info(f"Starting model calibration from {start_date} to {end_date}")
        
        # Step 1: Fetch real economic data
        self.raw_data = self.fred_client.get_core_economic_data(
            start_date=start_date,
            end_date=end_date
        )
        
        if not self.raw_data:
            raise ValueError("No economic data could be fetched from FRED")
        
        # Step 2: Process the data
        self.processed_data = self.data_processor.process_fred_data(self.raw_data)
        
        # Step 3: Split data for calibration and validation
        calibration_data, validation_data = self._split_data_for_validation(
            self.processed_data, validation_split
        )
        
        # Step 4: Calibrate parameters
        self.calibrated_params = self.data_processor.calibrate_simulation_parameters(
            calibration_data
        )
        
        # Step 5: Create validation targets
        self.validation_targets = self.data_processor.create_validation_targets(
            validation_data
        )
        
        # Step 6: Add real data context
        self.calibrated_params['real_data_context'] = {
            'calibration_period': f"{start_date} to {end_date}",
            'data_sources': list(self.raw_data.keys()),
            'calibration_date': datetime.now().isoformat(),
            'validation_targets': self.validation_targets
        }
        
        logger.info("Model calibration completed successfully")
        return self.calibrated_params
    
    def calibrate_for_scenario(
        self,
        scenario: str = "post_covid",
        custom_period: Optional[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Calibrate model for specific economic scenarios.
        
        Args:
            scenario: Predefined scenario ('great_recession', 'post_covid', 'stable_growth')
            custom_period: Custom (start_date, end_date) tuple
            
        Returns:
            Dictionary of scenario-specific calibrated parameters
        """
        # Define scenario periods
        scenarios = {
            'great_recession': ('2007-01-01', '2010-12-31'),
            'post_covid': ('2020-01-01', '2023-12-31'),
            'stable_growth': ('2012-01-01', '2019-12-31'),
            'full_history': ('2000-01-01', None)
        }
        
        if custom_period:
            start_date, end_date = custom_period
        elif scenario in scenarios:
            start_date, end_date = scenarios[scenario]
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        logger.info(f"Calibrating for scenario: {scenario}")
        
        # Calibrate for the specific period
        params = self.calibrate_from_period(start_date, end_date)
        
        # Add scenario-specific adjustments
        params['scenario'] = scenario
        params['scenario_adjustments'] = self._get_scenario_adjustments(scenario)
        
        return params
    
    def validate_calibration(
        self,
        simulation_results: pd.DataFrame,
        validation_metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Validate calibrated model against real data.
        
        Args:
            simulation_results: Results from calibrated simulation
            validation_metrics: Metrics to validate against
            
        Returns:
            Dictionary of validation scores
        """
        if not self.validation_targets:
            raise ValueError("No validation targets available. Run calibration first.")
        
        if validation_metrics is None:
            validation_metrics = ['unemployment', 'cpi', 'fed_funds', 'real_gdp']
        
        validation_scores = {}
        
        for metric in validation_metrics:
            if metric not in self.validation_targets:
                continue
            
            target = self.validation_targets[metric]
            
            if metric in simulation_results.columns:
                sim_values = simulation_results[metric]
                
                # Calculate validation scores
                mean_error = abs(sim_values.mean() - target['mean']) / target['mean']
                std_error = abs(sim_values.std() - target['std']) / target['std']
                trend_error = abs(
                    self.data_processor._calculate_trend(sim_values) - target['trend']
                )
                
                # Combined score (lower is better)
                combined_score = (mean_error + std_error + abs(trend_error)) / 3
                
                validation_scores[metric] = {
                    'mean_error': mean_error,
                    'std_error': std_error,
                    'trend_error': trend_error,
                    'combined_score': combined_score
                }
        
        # Overall validation score
        if validation_scores:
            overall_score = np.mean([
                scores['combined_score'] for scores in validation_scores.values()
            ])
            validation_scores['overall'] = overall_score
        
        return validation_scores
    
    def create_calibration_report(self, output_path: Optional[str] = None) -> str:
        """
        Create a comprehensive calibration report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Report as string
        """
        if not self.calibrated_params:
            return "No calibration data available. Run calibration first."
        
        report = []
        report.append("=" * 60)
        report.append("ECONAGENT-LIGHT MODEL CALIBRATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        if self.raw_data:
            report.append("DATA SOURCES:")
            report.append("-" * 20)
            for name, df in self.raw_data.items():
                series_id = self.fred_client.CORE_SERIES.get(name, 'Unknown')
                report.append(f"• {name.upper()}: {series_id} ({len(df)} observations)")
            report.append("")
        
        # Calibrated parameters
        report.append("CALIBRATED PARAMETERS:")
        report.append("-" * 25)
        
        key_params = [
            ('Productivity', 'productivity'),
            ('Max Price Inflation', 'max_price_inflation'),
            ('Max Wage Inflation', 'max_wage_inflation'),
            ('Base Interest Rate', 'base_interest_rate'),
            ('Natural Unemployment', 'natural_unemployment'),
            ('Phillips Curve Coeff', 'phillips_curve_coefficient'),
            ('Okun\'s Law Coeff', 'okuns_law_coefficient')
        ]
        
        for label, key in key_params:
            if key in self.calibrated_params:
                value = self.calibrated_params[key]
                report.append(f"• {label:<20}: {value:>8.4f}")
        
        report.append("")
        
        # Historical statistics
        if 'historical_stats' in self.calibrated_params:
            report.append("HISTORICAL STATISTICS:")
            report.append("-" * 25)
            
            for indicator, stats in self.calibrated_params['historical_stats'].items():
                report.append(f"\n{indicator.upper()}:")
                report.append(f"  Mean: {stats['mean']:>8.4f}")
                report.append(f"  Std:  {stats['std']:>8.4f}")
                report.append(f"  Min:  {stats['range'][0]:>8.4f}")
                report.append(f"  Max:  {stats['range'][1]:>8.4f}")
        
        # Validation targets
        if self.validation_targets:
            report.append("\nVALIDATION TARGETS:")
            report.append("-" * 20)
            for metric, targets in self.validation_targets.items():
                report.append(f"• {metric}: Mean={targets['mean']:.4f}, Std={targets['std']:.4f}")
        
        # Economic relationships
        report.append("\nECONOMIC RELATIONSHIPS:")
        report.append("-" * 25)
        
        if 'phillips_curve_coefficient' in self.calibrated_params:
            coeff = self.calibrated_params['phillips_curve_coefficient']
            report.append(f"• Phillips Curve: {coeff:.4f} (inflation vs unemployment)")
        
        if 'okuns_law_coefficient' in self.calibrated_params:
            coeff = self.calibrated_params['okuns_law_coefficient']
            report.append(f"• Okun's Law: {coeff:.4f} (GDP growth vs unemployment)")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Calibration report saved to {output_path}")
        
        return report_text
    
    def plot_calibration_data(self, output_dir: str = "./calibration_plots"):
        """
        Create plots showing the calibration data and relationships.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.processed_data:
            logger.warning("No processed data available for plotting")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Plot 1: Time series of key indicators
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Key Economic Indicators (FRED Data)', fontsize=16)
        
        indicators = ['unemployment', 'cpi', 'fed_funds', 'real_gdp']
        titles = ['Unemployment Rate (%)', 'Consumer Price Index', 'Federal Funds Rate (%)', 'Real GDP']
        
        for i, (indicator, title) in enumerate(zip(indicators, titles)):
            if indicator in self.processed_data:
                ax = axes[i//2, i%2]
                data = self.processed_data[indicator]['data']
                ax.plot(data.index, data.values, linewidth=2)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'economic_indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Phillips Curve
        if 'unemployment' in self.processed_data and 'cpi' in self.processed_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            unemployment = self.processed_data['unemployment']['data']
            cpi = self.processed_data['cpi']['data']
            inflation = cpi.pct_change(12) * 100  # Year-over-year inflation
            
            # Align data
            aligned = pd.concat([unemployment, inflation], axis=1).dropna()
            
            if len(aligned) > 0:
                ax.scatter(aligned.iloc[:, 0], aligned.iloc[:, 1], alpha=0.6)
                ax.set_xlabel('Unemployment Rate (%)')
                ax.set_ylabel('Inflation Rate (%)')
                ax.set_title('Phillips Curve (Real Data)')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(aligned.iloc[:, 0], aligned.iloc[:, 1], 1)
                p = np.poly1d(z)
                ax.plot(aligned.iloc[:, 0], p(aligned.iloc[:, 0]), "r--", alpha=0.8)
                
                plt.savefig(output_path / 'phillips_curve.png', dpi=300, bbox_inches='tight')
            
            plt.close()
        
        # Plot 3: Okun's Law
        if 'unemployment' in self.processed_data and 'real_gdp' in self.processed_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            unemployment = self.processed_data['unemployment']['data']
            gdp = self.processed_data['real_gdp']['data']
            
            # Calculate changes
            unemployment_change = unemployment.diff()
            gdp_growth = gdp.pct_change(4) * 100  # Quarterly growth rate
            
            # Align data
            aligned = pd.concat([gdp_growth, unemployment_change], axis=1).dropna()
            
            if len(aligned) > 0:
                ax.scatter(aligned.iloc[:, 0], aligned.iloc[:, 1], alpha=0.6)
                ax.set_xlabel('GDP Growth Rate (%)')
                ax.set_ylabel('Change in Unemployment Rate (pp)')
                ax.set_title("Okun's Law (Real Data)")
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(aligned.iloc[:, 0], aligned.iloc[:, 1], 1)
                p = np.poly1d(z)
                ax.plot(aligned.iloc[:, 0], p(aligned.iloc[:, 0]), "r--", alpha=0.8)
                
                plt.savefig(output_path / 'okuns_law.png', dpi=300, bbox_inches='tight')
            
            plt.close()
        
        logger.info(f"Calibration plots saved to {output_path}")
    
    def _split_data_for_validation(
        self,
        processed_data: Dict[str, Any],
        validation_split: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split processed data into calibration and validation sets."""
        calibration_data = {}
        validation_data = {}
        
        for name, data_dict in processed_data.items():
            series = data_dict['data']
            split_point = int(len(series) * (1 - validation_split))
            
            # Split the series
            cal_series = series.iloc[:split_point]
            val_series = series.iloc[split_point:]
            
            # Create new data dictionaries
            calibration_data[name] = {
                **data_dict,
                'data': cal_series
            }
            
            validation_data[name] = {
                **data_dict,
                'data': val_series
            }
        
        return calibration_data, validation_data
    
    def _get_scenario_adjustments(self, scenario: str) -> Dict[str, float]:
        """Get scenario-specific parameter adjustments."""
        adjustments = {
            'great_recession': {
                'unemployment_volatility_multiplier': 2.0,
                'gdp_volatility_multiplier': 3.0,
                'financial_stress_factor': 1.5
            },
            'post_covid': {
                'inflation_volatility_multiplier': 1.5,
                'supply_chain_disruption': 1.2,
                'policy_response_factor': 2.0
            },
            'stable_growth': {
                'volatility_dampening': 0.8,
                'steady_state_bias': 1.1
            }
        }
        
        return adjustments.get(scenario, {})


# Example usage
if __name__ == "__main__":
    # Initialize calibrator
    calibrator = ModelCalibrator()
    
    try:
        # Calibrate for post-COVID period
        params = calibrator.calibrate_for_scenario('post_covid')
        
        # Generate report
        report = calibrator.create_calibration_report('./calibration_report.txt')
        print("Calibration Report:")
        print("=" * 50)
        print(report)
        
        # Create plots
        calibrator.plot_calibration_data('./plots')
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        print("Note: You may need a FRED API key for full functionality")