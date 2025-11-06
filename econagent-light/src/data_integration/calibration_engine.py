"""
Economic calibration engine for EconAgent-Light.
Calibrates simulation parameters using real FRED economic data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from .fred_client import FREDClient, EconomicSnapshot

logger = logging.getLogger(__name__)

@dataclass
class CalibrationConfig:
    """Configuration for economic calibration."""
    # Calibration periods
    historical_years: int = 5
    trend_analysis_months: int = 12
    
    # Economic parameter bounds
    min_unemployment_target: float = 2.0
    max_unemployment_target: float = 10.0
    min_inflation_target: float = 0.5
    max_inflation_target: float = 5.0
    min_interest_rate: float = 0.0
    max_interest_rate: float = 8.0
    
    # Calibration weights
    unemployment_weight: float = 0.3
    inflation_weight: float = 0.3
    wage_growth_weight: float = 0.2
    participation_weight: float = 0.2

@dataclass
class CalibrationResult:
    """Result of economic calibration process."""
    # Calibrated parameters
    unemployment_target: float
    inflation_target: float
    natural_interest_rate: float
    productivity_growth: float
    wage_adjustment_rate: float
    price_adjustment_rate: float
    
    # Calibration metadata
    calibration_date: datetime
    data_period: str
    confidence_score: float
    fred_snapshot: EconomicSnapshot
    
    # Validation metrics
    historical_fit_score: float
    trend_alignment_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['calibration_date'] = self.calibration_date.isoformat()
        result['fred_snapshot'] = self.fred_snapshot.to_dict()
        return result

class CalibrationEngine:
    """
    Engine for calibrating economic simulation parameters using FRED data.
    """
    
    def __init__(
        self,
        fred_client: FREDClient,
        config: Optional[CalibrationConfig] = None
    ):
        """
        Initialize calibration engine.
        
        Args:
            fred_client: FRED API client for data fetching
            config: Calibration configuration (uses defaults if None)
        """
        self.fred_client = fred_client
        self.config = config or CalibrationConfig()
        
        # Cache for calibration results
        self.last_calibration: Optional[CalibrationResult] = None
        self.calibration_cache: Dict[str, CalibrationResult] = {}
    
    def calibrate_simulation_parameters(
        self,
        base_config: Optional[Dict[str, Any]] = None,
        force_recalibrate: bool = False
    ) -> CalibrationResult:
        """
        Calibrate simulation parameters using current FRED data.
        
        Args:
            base_config: Base simulation configuration to adjust
            force_recalibrate: Force recalibration even if recent results exist
            
        Returns:
            CalibrationResult with adjusted parameters
        """
        logger.info("Starting economic parameter calibration using FRED data")
        
        # Check if we have recent calibration results
        cache_key = datetime.now().strftime("%Y-%m-%d")
        if not force_recalibrate and cache_key in self.calibration_cache:
            logger.info("Using cached calibration results from today")
            return self.calibration_cache[cache_key]
        
        try:
            # Get current economic snapshot
            current_snapshot = self.fred_client.get_current_economic_snapshot()
            
            # Fetch historical data for calibration
            historical_data = self._fetch_historical_data()
            
            # Analyze economic trends
            trend_analysis = self._analyze_economic_trends(historical_data)
            
            # Calibrate parameters
            calibrated_params = self._calibrate_parameters(
                current_snapshot, 
                historical_data, 
                trend_analysis,
                base_config
            )
            
            # Validate calibration
            validation_scores = self._validate_calibration(
                calibrated_params, 
                historical_data
            )
            
            # Create calibration result
            result = CalibrationResult(
                unemployment_target=calibrated_params['unemployment_target'],
                inflation_target=calibrated_params['inflation_target'],
                natural_interest_rate=calibrated_params['natural_interest_rate'],
                productivity_growth=calibrated_params['productivity_growth'],
                wage_adjustment_rate=calibrated_params['wage_adjustment_rate'],
                price_adjustment_rate=calibrated_params['price_adjustment_rate'],
                calibration_date=datetime.now(),
                data_period=f"{self.config.historical_years} years",
                confidence_score=validation_scores['confidence'],
                fred_snapshot=current_snapshot,
                historical_fit_score=validation_scores['historical_fit'],
                trend_alignment_score=validation_scores['trend_alignment']
            )
            
            # Cache result
            self.calibration_cache[cache_key] = result
            self.last_calibration = result
            
            logger.info(f"Calibration completed with confidence score: {result.confidence_score:.2f}")
            logger.info(f"Key parameters: unemployment_target={result.unemployment_target:.1f}%, "
                       f"inflation_target={result.inflation_target:.1f}%, "
                       f"natural_rate={result.natural_interest_rate:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            # Return default calibration
            return self._get_default_calibration(current_snapshot)
    
    def _fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical economic data for calibration."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.config.historical_years * 365)).strftime("%Y-%m-%d")
        
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        
        # Key series for calibration
        series_to_fetch = [
            'UNRATE',      # Unemployment Rate
            'CPIAUCSL',    # Consumer Price Index
            'FEDFUNDS',    # Federal Funds Rate
            'AHETPI',      # Average Hourly Earnings
            'CIVPART',     # Labor Force Participation
            'GDPC1',       # Real GDP (Quarterly)
            'PAYEMS'       # Total Nonfarm Employment
        ]
        
        historical_data = {}
        
        for series_id in series_to_fetch:
            try:
                df = self.fred_client.get_series(series_id, start_date=start_date, end_date=end_date)
                if not df.empty:
                    historical_data[series_id] = df
                    logger.debug(f"Fetched {len(df)} observations for {series_id}")
                else:
                    logger.warning(f"No data available for {series_id}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
        
        return historical_data
    
    def _analyze_economic_trends(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze economic trends from historical data."""
        trends = {}
        
        try:
            # Unemployment trend
            if 'UNRATE' in historical_data:
                unemployment = historical_data['UNRATE']['UNRATE']
                trends['unemployment_trend'] = unemployment.rolling(window=12).mean().iloc[-1]
                trends['unemployment_volatility'] = unemployment.rolling(window=12).std().iloc[-1]
                trends['unemployment_recent_change'] = unemployment.iloc[-1] - unemployment.iloc[-12] if len(unemployment) >= 12 else 0
            
            # Inflation trend (CPI year-over-year)
            if 'CPIAUCSL' in historical_data:
                cpi = historical_data['CPIAUCSL']['CPIAUCSL']
                inflation_rates = cpi.pct_change(periods=12) * 100  # Year-over-year
                trends['inflation_trend'] = inflation_rates.rolling(window=6).mean().iloc[-1]
                trends['inflation_volatility'] = inflation_rates.rolling(window=12).std().iloc[-1]
                trends['inflation_recent_change'] = inflation_rates.iloc[-1] - inflation_rates.iloc[-6] if len(inflation_rates) >= 6 else 0
            
            # Interest rate trend
            if 'FEDFUNDS' in historical_data:
                fed_funds = historical_data['FEDFUNDS']['FEDFUNDS']
                trends['interest_rate_trend'] = fed_funds.rolling(window=6).mean().iloc[-1]
                trends['interest_rate_volatility'] = fed_funds.rolling(window=12).std().iloc[-1]
            
            # Wage growth trend
            if 'AHETPI' in historical_data:
                wages = historical_data['AHETPI']['AHETPI']
                wage_growth = wages.pct_change(periods=12) * 100  # Year-over-year
                trends['wage_growth_trend'] = wage_growth.rolling(window=6).mean().iloc[-1]
                trends['wage_growth_volatility'] = wage_growth.rolling(window=12).std().iloc[-1]
            
            # Labor participation trend
            if 'CIVPART' in historical_data:
                participation = historical_data['CIVPART']['CIVPART']
                trends['participation_trend'] = participation.rolling(window=12).mean().iloc[-1]
                trends['participation_change'] = participation.iloc[-1] - participation.iloc[-12] if len(participation) >= 12 else 0
            
            logger.info("Economic trend analysis completed")
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            # Return default trends
            trends = {
                'unemployment_trend': 5.0,
                'inflation_trend': 2.0,
                'interest_rate_trend': 2.5,
                'wage_growth_trend': 3.0,
                'participation_trend': 63.0
            }
        
        return trends
    
    def _calibrate_parameters(
        self,
        current_snapshot: EconomicSnapshot,
        historical_data: Dict[str, pd.DataFrame],
        trend_analysis: Dict[str, Any],
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Calibrate simulation parameters based on economic data."""
        
        # Start with default parameters
        params = {
            'unemployment_target': 4.0,
            'inflation_target': 2.0,
            'natural_interest_rate': 2.5,
            'productivity_growth': 1.5,
            'wage_adjustment_rate': 0.05,
            'price_adjustment_rate': 0.10
        }
        
        # Apply base configuration if provided
        if base_config:
            params.update(base_config)
        
        # Calibrate unemployment target
        if 'unemployment_trend' in trend_analysis:
            unemployment_target = trend_analysis['unemployment_trend']
            # Bound the target
            unemployment_target = max(self.config.min_unemployment_target, 
                                    min(self.config.max_unemployment_target, unemployment_target))
            params['unemployment_target'] = unemployment_target
        
        # Calibrate inflation target
        if 'inflation_trend' in trend_analysis:
            inflation_target = trend_analysis['inflation_trend']
            # Bound the target
            inflation_target = max(self.config.min_inflation_target,
                                 min(self.config.max_inflation_target, inflation_target))
            params['inflation_target'] = inflation_target
        
        # Calibrate natural interest rate (based on recent Fed policy)
        if 'interest_rate_trend' in trend_analysis:
            natural_rate = trend_analysis['interest_rate_trend']
            # Adjust for economic conditions
            if current_snapshot.unemployment_rate > params['unemployment_target']:
                natural_rate *= 0.9  # Lower rate in high unemployment
            if current_snapshot.inflation_rate > params['inflation_target']:
                natural_rate *= 1.1  # Higher rate in high inflation
            
            natural_rate = max(self.config.min_interest_rate,
                             min(self.config.max_interest_rate, natural_rate))
            params['natural_interest_rate'] = natural_rate
        
        # Calibrate productivity growth (based on wage trends)
        if 'wage_growth_trend' in trend_analysis:
            # Productivity growth typically tracks wage growth minus inflation
            productivity_growth = max(0.5, trend_analysis['wage_growth_trend'] - params['inflation_target'])
            params['productivity_growth'] = min(3.0, productivity_growth)
        
        # Calibrate adjustment rates based on volatility
        if 'unemployment_volatility' in trend_analysis and 'inflation_volatility' in trend_analysis:
            # Higher volatility suggests faster adjustment mechanisms
            unemployment_vol = trend_analysis.get('unemployment_volatility', 1.0)
            inflation_vol = trend_analysis.get('inflation_volatility', 1.0)
            
            # Scale adjustment rates based on observed volatility
            wage_adjustment = 0.05 * (1 + unemployment_vol / 2.0)
            price_adjustment = 0.10 * (1 + inflation_vol / 2.0)
            
            params['wage_adjustment_rate'] = min(0.15, wage_adjustment)
            params['price_adjustment_rate'] = min(0.25, price_adjustment)
        
        logger.info("Parameter calibration completed")
        return params
    
    def _validate_calibration(
        self,
        calibrated_params: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Validate calibration results against historical data."""
        
        validation_scores = {
            'confidence': 0.8,  # Default confidence
            'historical_fit': 0.7,
            'trend_alignment': 0.8
        }
        
        try:
            # Check parameter reasonableness
            param_score = 1.0
            
            # Unemployment target should be reasonable
            if not (2.0 <= calibrated_params['unemployment_target'] <= 8.0):
                param_score *= 0.8
            
            # Inflation target should be reasonable
            if not (1.0 <= calibrated_params['inflation_target'] <= 4.0):
                param_score *= 0.8
            
            # Interest rate should be reasonable
            if not (0.0 <= calibrated_params['natural_interest_rate'] <= 6.0):
                param_score *= 0.8
            
            validation_scores['confidence'] = param_score
            
            # Historical fit validation (simplified)
            if 'UNRATE' in historical_data and 'CPIAUCSL' in historical_data:
                unemployment_data = historical_data['UNRATE']['UNRATE']
                
                # Check if calibrated unemployment target is close to recent average
                recent_avg_unemployment = unemployment_data.tail(12).mean()
                unemployment_diff = abs(calibrated_params['unemployment_target'] - recent_avg_unemployment)
                
                # Score based on how close we are (within 2 percentage points is good)
                fit_score = max(0.0, 1.0 - unemployment_diff / 2.0)
                validation_scores['historical_fit'] = fit_score
            
            logger.info(f"Calibration validation completed: confidence={validation_scores['confidence']:.2f}")
            
        except Exception as e:
            logger.warning(f"Calibration validation failed: {e}")
        
        return validation_scores
    
    def _get_default_calibration(self, snapshot: EconomicSnapshot) -> CalibrationResult:
        """Get default calibration when data-driven calibration fails."""
        logger.warning("Using default calibration parameters")
        
        return CalibrationResult(
            unemployment_target=4.0,
            inflation_target=2.0,
            natural_interest_rate=2.5,
            productivity_growth=1.5,
            wage_adjustment_rate=0.05,
            price_adjustment_rate=0.10,
            calibration_date=datetime.now(),
            data_period="default",
            confidence_score=0.5,
            fred_snapshot=snapshot,
            historical_fit_score=0.5,
            trend_alignment_score=0.5
        )
    
    def save_calibration_report(self, result: CalibrationResult, filepath: str) -> None:
        """Save calibration result to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Calibration report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save calibration report: {e}")
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of last calibration."""
        if not self.last_calibration:
            return {"status": "No calibration performed yet"}
        
        result = self.last_calibration
        return {
            "status": "Calibrated",
            "calibration_date": result.calibration_date.isoformat(),
            "confidence_score": result.confidence_score,
            "parameters": {
                "unemployment_target": result.unemployment_target,
                "inflation_target": result.inflation_target,
                "natural_interest_rate": result.natural_interest_rate,
                "productivity_growth": result.productivity_growth
            },
            "current_conditions": {
                "unemployment_rate": result.fred_snapshot.unemployment_rate,
                "inflation_rate": result.fred_snapshot.inflation_rate,
                "fed_funds_rate": result.fred_snapshot.fed_funds_rate
            }
        }


# Example usage
if __name__ == "__main__":
    from fred_client import FREDClient
    
    # Initialize clients
    fred_client = FREDClient()
    calibration_engine = CalibrationEngine(fred_client)
    
    try:
        # Perform calibration
        result = calibration_engine.calibrate_simulation_parameters()
        
        print("Calibration Results:")
        print(f"  Unemployment Target: {result.unemployment_target:.1f}%")
        print(f"  Inflation Target: {result.inflation_target:.1f}%")
        print(f"  Natural Interest Rate: {result.natural_interest_rate:.1f}%")
        print(f"  Confidence Score: {result.confidence_score:.2f}")
        
        # Get summary
        summary = calibration_engine.get_calibration_summary()
        print(f"\nCalibration Summary: {summary}")
        
    except Exception as e:
        print(f"Calibration example failed: {e}")