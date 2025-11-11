"""
Risk Manager Agent - monitors risk signals and computes assessments.
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd

from src.agents.event_bus import EventBus
from src.agents.events import (
    DataUpdateEvent,
    ForecastCompleteEvent,
    SimulationCompleteEvent,
    KRIUpdateEvent,
    RiskSignalEvent,
    RiskLevel
)
from src.kri.calculator import KRICalculator


logger = logging.getLogger(__name__)


class RiskManagerAgent:
    """
    Event-driven agent that monitors risk signals and computes assessments.
    
    The Risk Manager subscribes to data updates, forecast completions, and
    simulation results, then computes KRIs and evaluates risk thresholds.
    """
    
    def __init__(
        self,
        kri_calculator: KRICalculator,
        event_bus: EventBus,
        agent_id: str = "RiskManager"
    ):
        """
        Initialize the Risk Manager agent.
        
        Args:
            kri_calculator: KRI calculator instance
            event_bus: Event bus for pub/sub communication
            agent_id: Unique identifier for this agent
        """
        self.kri_calculator = kri_calculator
        self.event_bus = event_bus
        self.agent_id = agent_id
        
        # State tracking
        self.current_kris: Dict[str, float] = {}
        self.current_alerts: Dict[str, RiskLevel] = {}
        self.latest_forecasts: Optional[pd.DataFrame] = None
        self.latest_simulation_results: Optional[pd.DataFrame] = None
        self.latest_data: Optional[pd.DataFrame] = None
        self.risk_signals: list = []
        
        # Subscribe to relevant events
        self._subscribe_to_events()
        
        logger.info(f"Risk Manager Agent '{agent_id}' initialized")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant event types."""
        self.event_bus.subscribe('DataUpdateEvent', self.on_data_update)
        self.event_bus.subscribe('ForecastCompleteEvent', self.on_forecast_complete)
        self.event_bus.subscribe('SimulationCompleteEvent', self.on_simulation_complete)
        self.event_bus.subscribe('RiskSignalEvent', self.on_risk_signal)
        
        logger.info(f"{self.agent_id} subscribed to events")
    
    def on_data_update(self, event: DataUpdateEvent):
        """
        Handle new data availability.
        
        Args:
            event: DataUpdateEvent with new data
        """
        logger.info(
            f"{self.agent_id} received data update for series: {event.series_ids}"
        )
        
        # Store latest data
        self.latest_data = event.data
        
        # Check if we should recalculate risk
        if self._should_recalculate_risk(event):
            logger.info(f"{self.agent_id} triggering risk recalculation")
            self._compute_and_publish_kris()
    
    def on_forecast_complete(self, event: ForecastCompleteEvent):
        """
        Handle new forecast availability.
        
        Args:
            event: ForecastCompleteEvent with forecast results
        """
        logger.info(
            f"{self.agent_id} received forecast for horizon {event.horizon} "
            f"from models: {event.model_names}"
        )
        
        # Store latest forecasts
        # Convert forecast results to DataFrame for KRI calculation
        self.latest_forecasts = self._extract_forecast_dataframe(event.forecasts)
        
        # Compute KRIs from forecasts
        self._compute_and_publish_kris()
    
    def on_simulation_complete(self, event: SimulationCompleteEvent):
        """
        Handle simulation completion.
        
        Args:
            event: SimulationCompleteEvent with simulation results
        """
        logger.info(
            f"{self.agent_id} received simulation results for scenario: {event.scenario_name}"
        )
        
        # Store latest simulation results
        self.latest_simulation_results = event.results
        
        # Recompute KRIs with new simulation data
        self._compute_and_publish_kris()
    
    def on_risk_signal(self, event: RiskSignalEvent):
        """
        Handle risk signals from other agents.
        
        Args:
            event: RiskSignalEvent with detected risk signals
        """
        logger.info(
            f"{self.agent_id} received {len(event.signals)} risk signals "
            f"from {event.source_agent}"
        )
        
        # Store risk signals
        self.risk_signals.extend(event.signals)
        
        # Log each signal
        for signal in event.signals:
            logger.warning(
                f"Risk Signal: {signal.type} (severity: {signal.severity}) - "
                f"{signal.description}"
            )
        
        # If high severity signals, trigger immediate KRI recalculation
        high_severity_signals = [
            s for s in event.signals 
            if s.severity in ['high', 'critical']
        ]
        
        if high_severity_signals:
            logger.warning(
                f"{self.agent_id} detected {len(high_severity_signals)} high-severity signals, "
                "triggering immediate risk assessment"
            )
            self._compute_and_publish_kris()
    
    def _should_recalculate_risk(self, event: DataUpdateEvent) -> bool:
        """
        Determine if risk should be recalculated based on data update.
        
        Args:
            event: DataUpdateEvent
            
        Returns:
            True if risk should be recalculated
        """
        # Recalculate if we have key economic indicators
        key_indicators = ['unemployment', 'interest_rate', 'inflation', 'gdp']
        
        has_key_indicator = any(
            indicator in series_id.lower() 
            for series_id in event.series_ids 
            for indicator in key_indicators
        )
        
        return has_key_indicator
    
    def _extract_forecast_dataframe(
        self, 
        forecasts: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Extract forecast data into DataFrame format for KRI calculation.
        
        Args:
            forecasts: Dictionary of forecast results
            
        Returns:
            DataFrame with forecast values
        """
        forecast_data = {}
        
        for kri_name, forecast_result in forecasts.items():
            if hasattr(forecast_result, 'point_forecast'):
                # Take the last forecast value
                forecast_data[kri_name] = [forecast_result.point_forecast[-1]]
        
        if forecast_data:
            return pd.DataFrame(forecast_data)
        else:
            return pd.DataFrame()
    
    def _compute_and_publish_kris(self):
        """
        Compute KRIs and publish update event.
        """
        try:
            # Compute all KRIs
            kris = self.kri_calculator.compute_all_kris(
                forecasts=self.latest_forecasts,
                simulation_results=self.latest_simulation_results,
                portfolio_data=None,  # Could be added later
                balance_sheet=None    # Could be added later
            )
            
            # Evaluate thresholds
            alerts = self.kri_calculator.evaluate_thresholds(kris)
            
            # Convert RiskLevel enum to string for alerts dict
            alerts_str = {k: v.value for k, v in alerts.items()}
            
            # Detect trends (if we have history)
            trends = self._detect_trends(kris)
            
            # Update internal state
            self.current_kris = kris
            self.current_alerts = alerts
            
            # Log critical alerts
            critical_kris = [
                kri_name for kri_name, level in alerts.items()
                if level == RiskLevel.CRITICAL
            ]
            
            if critical_kris:
                logger.critical(
                    f"{self.agent_id} detected CRITICAL risk levels for: {critical_kris}"
                )
            
            # Publish KRI update event
            kri_event = KRIUpdateEvent(
                kris=kris,
                alerts=alerts_str,
                timestamp=datetime.now(),
                risk_levels=alerts,
                trends=trends,
                metadata={
                    'agent_id': self.agent_id,
                    'has_forecasts': self.latest_forecasts is not None,
                    'has_simulation': self.latest_simulation_results is not None,
                    'n_risk_signals': len(self.risk_signals)
                }
            )
            
            self.event_bus.publish('KRIUpdateEvent', kri_event)
            
            logger.info(
                f"{self.agent_id} published KRI update with {len(kris)} indicators"
            )
            
        except Exception as e:
            logger.error(
                f"{self.agent_id} failed to compute KRIs: {e}",
                exc_info=True
            )
    
    def _detect_trends(self, current_kris: Dict[str, float]) -> Dict[str, str]:
        """
        Detect trends in KRI values.
        
        Args:
            current_kris: Current KRI values
            
        Returns:
            Dictionary of KRI names to trend descriptions
        """
        trends = {}
        
        # For now, return 'stable' for all
        # In a real implementation, we would track KRI history
        for kri_name in current_kris.keys():
            trends[kri_name] = 'stable'
        
        return trends
    
    def get_current_risk_summary(self) -> Dict[str, Any]:
        """
        Get current risk summary.
        
        Returns:
            Dictionary with current risk state
        """
        return {
            'kris': self.current_kris,
            'alerts': {k: v.value for k, v in self.current_alerts.items()},
            'n_critical': sum(
                1 for v in self.current_alerts.values() 
                if v == RiskLevel.CRITICAL
            ),
            'n_high': sum(
                1 for v in self.current_alerts.values() 
                if v == RiskLevel.HIGH
            ),
            'n_risk_signals': len(self.risk_signals),
            'has_forecasts': self.latest_forecasts is not None,
            'has_simulation': self.latest_simulation_results is not None
        }
    
    def reset_state(self):
        """Reset agent state."""
        self.current_kris = {}
        self.current_alerts = {}
        self.latest_forecasts = None
        self.latest_simulation_results = None
        self.latest_data = None
        self.risk_signals = []
        
        logger.info(f"{self.agent_id} state reset")
