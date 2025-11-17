"""
KRI Calculator for computing risk indicators from data and forecasts.
"""
from typing import Dict, Optional, List
import numpy as np
import pandas as pd

from src.kri.definitions import kri_registry, RiskLevel
from src.utils.logging_config import logger


class KRICalculator:
    """
    Computes Key Risk Indicators from forecasts and simulation results.
    """
    
    def __init__(self):
        """Initialize KRI calculator."""
        self.registry = kri_registry
        logger.info("KRI Calculator initialized")
    
    def compute_credit_kris(
        self,
        forecasts: Optional[pd.DataFrame] = None,
        simulation_results: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Compute credit risk KRIs.
        
        Args:
            forecasts: DataFrame with economic forecasts
            simulation_results: DataFrame with simulation outputs
            
        Returns:
            Dictionary of KRI names to values
        """
        kris = {}
        
        # Loan default rate (from simulation)
        if simulation_results is not None and 'default_rate' in simulation_results.columns:
            kris['loan_default_rate'] = float(simulation_results['default_rate'].mean())
        else:
            # Estimate from unemployment if available
            if forecasts is not None and 'unemployment' in forecasts.columns:
                unemployment = forecasts['unemployment'].iloc[-1]
                # Simple model: default rate increases with unemployment
                kris['loan_default_rate'] = max(0.02, (unemployment - 3.5) * 0.5 + 0.02)
        
        # Delinquency rate (leading indicator)
        if forecasts is not None and 'unemployment' in forecasts.columns:
            unemployment_forecast = forecasts['unemployment'].iloc[-1]
            kris['delinquency_rate'] = self._estimate_delinquency(unemployment_forecast)
        
        # Credit quality score (inverse relationship with unemployment)
        if forecasts is not None and 'unemployment' in forecasts.columns:
            unemployment = forecasts['unemployment'].iloc[-1]
            # Higher unemployment -> lower credit quality
            kris['credit_quality_score'] = max(550, 750 - (unemployment - 3.5) * 15)
        
        # Loan concentration (from simulation or default)
        if simulation_results is not None and 'loan_concentration' in simulation_results.columns:
            kris['loan_concentration'] = float(simulation_results['loan_concentration'].mean())
        else:
            kris['loan_concentration'] = 25.0  # Default moderate concentration
        
        logger.info(f"Computed {len(kris)} credit KRIs")
        return kris
    
    def compute_market_kris(
        self,
        forecasts: Optional[pd.DataFrame] = None,
        portfolio_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Compute market risk KRIs.
        
        Args:
            forecasts: DataFrame with economic forecasts
            portfolio_data: DataFrame with portfolio returns/values
            
        Returns:
            Dictionary of KRI names to values
        """
        kris = {}
        
        # Portfolio volatility
        if portfolio_data is not None and 'returns' in portfolio_data.columns:
            returns = portfolio_data['returns']
            kris['portfolio_volatility'] = float(returns.std() * np.sqrt(252) * 100)
        else:
            # Estimate from economic volatility
            if forecasts is not None:
                # Use volatility of forecasts as proxy
                vol = forecasts.std().mean()
                kris['portfolio_volatility'] = float(vol * 10)  # Scale up
        
        # Value at Risk (95% confidence)
        if portfolio_data is not None and 'returns' in portfolio_data.columns:
            returns = portfolio_data['returns']
            var_95 = np.percentile(returns, 5)
            if 'value' in portfolio_data.columns:
                portfolio_value = portfolio_data['value'].iloc[-1]
                kris['var_95'] = abs(float(var_95 * portfolio_value))
            else:
                kris['var_95'] = abs(float(var_95 * 100))
        else:
            kris['var_95'] = 2.5  # Default moderate VaR
        
        # Interest rate risk
        if forecasts is not None and 'interest_rate' in forecasts.columns:
            rate_volatility = forecasts['interest_rate'].std()
            kris['interest_rate_risk'] = float(rate_volatility * 2)  # Duration proxy
        else:
            kris['interest_rate_risk'] = 3.0  # Default
        
        logger.info(f"Computed {len(kris)} market KRIs")
        return kris
    
    def compute_liquidity_kris(
        self,
        balance_sheet: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Compute liquidity risk KRIs.
        
        Args:
            balance_sheet: DataFrame with balance sheet data
            
        Returns:
            Dictionary of KRI names to values
        """
        kris = {}
        
        if balance_sheet is not None:
            # Liquidity coverage ratio
            if 'cash' in balance_sheet.columns and 'deposits' in balance_sheet.columns:
                liquid_assets = balance_sheet['cash'].iloc[-1]
                if 'marketable_securities' in balance_sheet.columns:
                    liquid_assets += balance_sheet['marketable_securities'].iloc[-1]
                
                net_outflows = balance_sheet['deposits'].iloc[-1] * 0.1  # 10% runoff
                kris['liquidity_coverage_ratio'] = float(liquid_assets / net_outflows)
            
            # Deposit flow ratio
            if 'deposit_change' in balance_sheet.columns and 'deposits' in balance_sheet.columns:
                deposit_change = balance_sheet['deposit_change'].iloc[-1]
                deposits = balance_sheet['deposits'].iloc[-1]
                kris['deposit_flow_ratio'] = float((deposit_change / deposits) * 100)
        else:
            # Defaults
            kris['liquidity_coverage_ratio'] = 1.3
            kris['deposit_flow_ratio'] = -2.0
        
        logger.info(f"Computed {len(kris)} liquidity KRIs")
        return kris
    
    def compute_all_kris(
        self,
        forecasts: Optional[pd.DataFrame] = None,
        simulation_results: Optional[pd.DataFrame] = None,
        portfolio_data: Optional[pd.DataFrame] = None,
        balance_sheet: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Compute all KRIs across all categories.
        
        Returns:
            Dictionary of all KRI names to values
        """
        all_kris = {}
        
        all_kris.update(self.compute_credit_kris(forecasts, simulation_results))
        all_kris.update(self.compute_market_kris(forecasts, portfolio_data))
        all_kris.update(self.compute_liquidity_kris(balance_sheet))
        
        logger.info(f"Computed total of {len(all_kris)} KRIs")
        return all_kris
    
    def evaluate_thresholds(
        self,
        kris: Dict[str, float]
    ) -> Dict[str, RiskLevel]:
        """
        Evaluate KRIs against risk thresholds.
        
        Args:
            kris: Dictionary of KRI names to values
            
        Returns:
            Dictionary of KRI names to risk levels
        """
        alerts = {}
        
        for kri_name, kri_value in kris.items():
            kri_def = self.registry.get_kri(kri_name)
            
            if kri_def is None:
                logger.warning(f"Unknown KRI: {kri_name}")
                continue
            
            thresholds = kri_def.thresholds
            
            # Determine risk level based on thresholds
            # Note: Some KRIs are "lower is better" (e.g., default rate)
            # Others are "higher is better" (e.g., liquidity ratio)
            # Special handling for negative thresholds (e.g., deposit_flow_ratio)
            
            if kri_name in ['liquidity_coverage_ratio', 'credit_quality_score']:
                # Higher is better
                if kri_value >= thresholds['low']:
                    alerts[kri_name] = RiskLevel.LOW
                elif kri_value >= thresholds['medium']:
                    alerts[kri_name] = RiskLevel.MEDIUM
                elif kri_value >= thresholds['high']:
                    alerts[kri_name] = RiskLevel.HIGH
                else:
                    alerts[kri_name] = RiskLevel.CRITICAL
            elif all(v < 0 for v in thresholds.values()):
                # Negative thresholds (e.g., deposit_flow_ratio: -5, -10, -15, -25)
                # More negative = worse, so -2 is better than -5
                if kri_value >= thresholds['low']:  # e.g., -2 >= -5 is True = LOW
                    alerts[kri_name] = RiskLevel.LOW
                elif kri_value >= thresholds['medium']:  # e.g., -7 >= -10 is True = MEDIUM
                    alerts[kri_name] = RiskLevel.MEDIUM
                elif kri_value >= thresholds['high']:
                    alerts[kri_name] = RiskLevel.HIGH
                else:
                    alerts[kri_name] = RiskLevel.CRITICAL
            else:
                # Lower is better (most KRIs) - positive thresholds
                if kri_value <= thresholds['low']:
                    alerts[kri_name] = RiskLevel.LOW
                elif kri_value <= thresholds['medium']:
                    alerts[kri_name] = RiskLevel.MEDIUM
                elif kri_value <= thresholds['high']:
                    alerts[kri_name] = RiskLevel.HIGH
                else:
                    alerts[kri_name] = RiskLevel.CRITICAL
        
        # Count alerts by level
        level_counts = {}
        for level in RiskLevel:
            level_counts[level.value] = sum(1 for v in alerts.values() if v == level)
        
        logger.info(f"Risk levels: {level_counts}")
        
        return alerts
    
    def detect_trend(
        self,
        kri_history: pd.Series
    ) -> str:
        """
        Detect trend in KRI values.
        
        Args:
            kri_history: Time series of KRI values
            
        Returns:
            Trend description: 'improving', 'stable', or 'deteriorating'
        """
        if len(kri_history) < 3:
            return 'stable'
        
        recent = kri_history.iloc[-3:]
        
        # Calculate trend
        if recent.iloc[-1] < recent.iloc[0] * 0.95:
            return 'improving'  # Assuming lower is better
        elif recent.iloc[-1] > recent.iloc[0] * 1.05:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _estimate_delinquency(self, unemployment_rate: float) -> float:
        """
        Estimate delinquency rate from unemployment.
        
        Args:
            unemployment_rate: Unemployment rate as percentage
            
        Returns:
            Estimated delinquency rate
        """
        # Simple linear model: delinquency increases with unemployment
        # Base rate of 3% at 4% unemployment, increases 0.6% per 1% unemployment
        base_rate = 3.0
        sensitivity = 0.6
        baseline_unemployment = 4.0
        
        delinquency = base_rate + (unemployment_rate - baseline_unemployment) * sensitivity
        return max(1.0, delinquency)  # Floor at 1%
