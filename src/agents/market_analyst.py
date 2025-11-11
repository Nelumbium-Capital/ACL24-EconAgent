"""
Market Analyst Agent - analyzes market data and identifies risk signals.
"""
import logging
from typing import List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from src.agents.event_bus import EventBus
from src.agents.events import (
    DataUpdateEvent,
    RiskSignalEvent,
    RiskSignal
)


logger = logging.getLogger(__name__)


class MarketAnalystAgent:
    """
    Analyzes market data and identifies risk signals.
    
    The Market Analyst monitors incoming data for anomalies such as:
    - Volatility spikes
    - Credit spread widening
    - Unusual market movements
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        agent_id: str = "MarketAnalyst",
        volatility_threshold: float = 2.0,
        spread_threshold: float = 0.5
    ):
        """
        Initialize the Market Analyst agent.
        
        Args:
            event_bus: Event bus for pub/sub communication
            agent_id: Unique identifier for this agent
            volatility_threshold: Number of standard deviations for volatility spike detection
            spread_threshold: Percentage point change for spread widening detection
        """
        self.event_bus = event_bus
        self.agent_id = agent_id
        self.volatility_threshold = volatility_threshold
        self.spread_threshold = spread_threshold
        
        # Historical data tracking
        self.data_history: List[pd.DataFrame] = []
        self.max_history_length = 100
        
        # Subscribe to data update events
        self._subscribe_to_events()
        
        logger.info(
            f"Market Analyst Agent '{agent_id}' initialized "
            f"(vol_threshold={volatility_threshold}, spread_threshold={spread_threshold})"
        )
    
    def _subscribe_to_events(self):
        """Subscribe to relevant event types."""
        self.event_bus.subscribe('DataUpdateEvent', self.on_data_update)
        logger.info(f"{self.agent_id} subscribed to DataUpdateEvent")
    
    def on_data_update(self, event: DataUpdateEvent):
        """
        Analyze new market data for risk signals.
        
        Args:
            event: DataUpdateEvent with new data
        """
        logger.info(
            f"{self.agent_id} analyzing data update for series: {event.series_ids}"
        )
        
        # Store data in history
        self._update_history(event.data)
        
        # Detect risk signals
        signals = []
        
        # Check for volatility spikes
        volatility_signals = self._detect_volatility_spike(event.data)
        signals.extend(volatility_signals)
        
        # Check for credit spread widening
        spread_signals = self._detect_spread_widening(event.data)
        signals.extend(spread_signals)
        
        # Check for interest rate shocks
        rate_signals = self._detect_interest_rate_shock(event.data)
        signals.extend(rate_signals)
        
        # Check for unemployment spikes
        unemployment_signals = self._detect_unemployment_spike(event.data)
        signals.extend(unemployment_signals)
        
        # Publish signals if any detected
        if signals:
            logger.warning(
                f"{self.agent_id} detected {len(signals)} risk signals"
            )
            
            risk_event = RiskSignalEvent(
                signals=signals,
                timestamp=datetime.now(),
                source_agent=self.agent_id,
                metadata={
                    'n_signals': len(signals),
                    'series_analyzed': event.series_ids
                }
            )
            
            self.event_bus.publish('RiskSignalEvent', risk_event)
        else:
            logger.debug(f"{self.agent_id} detected no risk signals")
    
    def _update_history(self, data: pd.DataFrame):
        """
        Update historical data tracking.
        
        Args:
            data: New data to add to history
        """
        self.data_history.append(data.copy())
        
        # Limit history length
        if len(self.data_history) > self.max_history_length:
            self.data_history = self.data_history[-self.max_history_length:]
    
    def _detect_volatility_spike(self, data: pd.DataFrame) -> List[RiskSignal]:
        """
        Detect abnormal volatility increases.
        
        Args:
            data: Current data
            
        Returns:
            List of risk signals for volatility spikes
        """
        signals = []
        
        # Need sufficient history for volatility calculation
        if len(self.data_history) < 20:
            return signals
        
        # Combine historical data
        try:
            historical_data = pd.concat(self.data_history, ignore_index=True)
            
            # Check numeric columns for volatility
            numeric_cols = historical_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col in data.columns and len(historical_data[col].dropna()) >= 20:
                    # Calculate rolling volatility
                    recent_vol = historical_data[col].tail(20).std()
                    historical_vol = historical_data[col].std()
                    
                    # Check if recent volatility exceeds threshold
                    if recent_vol > historical_vol * self.volatility_threshold:
                        severity = self._calculate_severity(
                            recent_vol / historical_vol,
                            thresholds={'medium': 1.5, 'high': 2.0, 'critical': 3.0}
                        )
                        
                        signals.append(RiskSignal(
                            type='volatility_spike',
                            severity=severity,
                            description=(
                                f"Volatility spike detected in {col}: "
                                f"recent volatility {recent_vol:.2f} is "
                                f"{recent_vol/historical_vol:.1f}x historical average"
                            ),
                            timestamp=datetime.now(),
                            affected_kris=['portfolio_volatility', 'var_95'],
                            metadata={
                                'series': col,
                                'recent_vol': float(recent_vol),
                                'historical_vol': float(historical_vol),
                                'ratio': float(recent_vol / historical_vol)
                            }
                        ))
                        
                        logger.warning(
                            f"Volatility spike in {col}: {recent_vol:.2f} "
                            f"(historical: {historical_vol:.2f})"
                        )
        
        except Exception as e:
            logger.error(f"Error detecting volatility spike: {e}", exc_info=True)
        
        return signals
    
    def _detect_spread_widening(self, data: pd.DataFrame) -> List[RiskSignal]:
        """
        Detect credit spread widening.
        
        Args:
            data: Current data
            
        Returns:
            List of risk signals for spread widening
        """
        signals = []
        
        # Look for credit spread indicators
        spread_columns = [
            col for col in data.columns 
            if 'spread' in col.lower() or 'baa' in col.lower()
        ]
        
        if not spread_columns or len(self.data_history) < 2:
            return signals
        
        try:
            # Get previous data
            prev_data = self.data_history[-2] if len(self.data_history) >= 2 else None
            
            if prev_data is not None:
                for col in spread_columns:
                    if col in data.columns and col in prev_data.columns:
                        current_spread = data[col].iloc[-1]
                        prev_spread = prev_data[col].iloc[-1]
                        
                        # Check for widening
                        spread_change = current_spread - prev_spread
                        
                        if spread_change > self.spread_threshold:
                            severity = self._calculate_severity(
                                spread_change,
                                thresholds={'medium': 0.3, 'high': 0.5, 'critical': 1.0}
                            )
                            
                            signals.append(RiskSignal(
                                type='credit_spread_widening',
                                severity=severity,
                                description=(
                                    f"Credit spread widening detected in {col}: "
                                    f"increased by {spread_change:.2f} percentage points "
                                    f"to {current_spread:.2f}%"
                                ),
                                timestamp=datetime.now(),
                                affected_kris=['loan_default_rate', 'credit_quality_score'],
                                metadata={
                                    'series': col,
                                    'current_spread': float(current_spread),
                                    'previous_spread': float(prev_spread),
                                    'change': float(spread_change)
                                }
                            ))
                            
                            logger.warning(
                                f"Credit spread widening in {col}: "
                                f"{prev_spread:.2f}% -> {current_spread:.2f}%"
                            )
        
        except Exception as e:
            logger.error(f"Error detecting spread widening: {e}", exc_info=True)
        
        return signals
    
    def _detect_interest_rate_shock(self, data: pd.DataFrame) -> List[RiskSignal]:
        """
        Detect sudden interest rate changes.
        
        Args:
            data: Current data
            
        Returns:
            List of risk signals for interest rate shocks
        """
        signals = []
        
        # Look for interest rate columns
        rate_columns = [
            col for col in data.columns 
            if 'rate' in col.lower() or 'fedfunds' in col.lower()
        ]
        
        if not rate_columns or len(self.data_history) < 2:
            return signals
        
        try:
            prev_data = self.data_history[-2] if len(self.data_history) >= 2 else None
            
            if prev_data is not None:
                for col in rate_columns:
                    if col in data.columns and col in prev_data.columns:
                        current_rate = data[col].iloc[-1]
                        prev_rate = prev_data[col].iloc[-1]
                        
                        rate_change = abs(current_rate - prev_rate)
                        
                        # Threshold for "shock" is 0.5 percentage points
                        if rate_change > 0.5:
                            severity = self._calculate_severity(
                                rate_change,
                                thresholds={'medium': 0.5, 'high': 1.0, 'critical': 2.0}
                            )
                            
                            signals.append(RiskSignal(
                                type='interest_rate_shock',
                                severity=severity,
                                description=(
                                    f"Interest rate shock detected in {col}: "
                                    f"changed by {rate_change:.2f} percentage points "
                                    f"to {current_rate:.2f}%"
                                ),
                                timestamp=datetime.now(),
                                affected_kris=['interest_rate_risk', 'var_95'],
                                metadata={
                                    'series': col,
                                    'current_rate': float(current_rate),
                                    'previous_rate': float(prev_rate),
                                    'change': float(rate_change)
                                }
                            ))
                            
                            logger.warning(
                                f"Interest rate shock in {col}: "
                                f"{prev_rate:.2f}% -> {current_rate:.2f}%"
                            )
        
        except Exception as e:
            logger.error(f"Error detecting interest rate shock: {e}", exc_info=True)
        
        return signals
    
    def _detect_unemployment_spike(self, data: pd.DataFrame) -> List[RiskSignal]:
        """
        Detect sudden unemployment increases.
        
        Args:
            data: Current data
            
        Returns:
            List of risk signals for unemployment spikes
        """
        signals = []
        
        # Look for unemployment columns
        unemployment_columns = [
            col for col in data.columns 
            if 'unemployment' in col.lower() or 'unrate' in col.lower()
        ]
        
        if not unemployment_columns or len(self.data_history) < 2:
            return signals
        
        try:
            prev_data = self.data_history[-2] if len(self.data_history) >= 2 else None
            
            if prev_data is not None:
                for col in unemployment_columns:
                    if col in data.columns and col in prev_data.columns:
                        current_rate = data[col].iloc[-1]
                        prev_rate = prev_data[col].iloc[-1]
                        
                        rate_change = current_rate - prev_rate
                        
                        # Threshold for spike is 0.5 percentage points increase
                        if rate_change > 0.5:
                            severity = self._calculate_severity(
                                rate_change,
                                thresholds={'medium': 0.5, 'high': 1.0, 'critical': 2.0}
                            )
                            
                            signals.append(RiskSignal(
                                type='unemployment_spike',
                                severity=severity,
                                description=(
                                    f"Unemployment spike detected in {col}: "
                                    f"increased by {rate_change:.2f} percentage points "
                                    f"to {current_rate:.2f}%"
                                ),
                                timestamp=datetime.now(),
                                affected_kris=['loan_default_rate', 'delinquency_rate'],
                                metadata={
                                    'series': col,
                                    'current_rate': float(current_rate),
                                    'previous_rate': float(prev_rate),
                                    'change': float(rate_change)
                                }
                            ))
                            
                            logger.warning(
                                f"Unemployment spike in {col}: "
                                f"{prev_rate:.2f}% -> {current_rate:.2f}%"
                            )
        
        except Exception as e:
            logger.error(f"Error detecting unemployment spike: {e}", exc_info=True)
        
        return signals
    
    def _calculate_severity(
        self, 
        value: float, 
        thresholds: dict
    ) -> str:
        """
        Calculate severity level based on value and thresholds.
        
        Args:
            value: Value to evaluate
            thresholds: Dictionary with 'medium', 'high', 'critical' thresholds
            
        Returns:
            Severity level: 'low', 'medium', 'high', or 'critical'
        """
        if value >= thresholds.get('critical', float('inf')):
            return 'critical'
        elif value >= thresholds.get('high', float('inf')):
            return 'high'
        elif value >= thresholds.get('medium', float('inf')):
            return 'medium'
        else:
            return 'low'
    
    def get_analysis_summary(self) -> dict:
        """
        Get summary of market analysis.
        
        Returns:
            Dictionary with analysis statistics
        """
        return {
            'agent_id': self.agent_id,
            'history_length': len(self.data_history),
            'volatility_threshold': self.volatility_threshold,
            'spread_threshold': self.spread_threshold
        }
    
    def reset_history(self):
        """Reset historical data tracking."""
        self.data_history = []
        logger.info(f"{self.agent_id} history reset")
