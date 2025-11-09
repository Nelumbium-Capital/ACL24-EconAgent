"""
Demonstration of the event-driven agent architecture.

This script shows how the EventBus, RiskManagerAgent, and MarketAnalystAgent
work together to process data updates and generate risk assessments.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.agents import (
    EventBus,
    RiskManagerAgent,
    MarketAnalystAgent,
    DataUpdateEvent
)
from src.kri.calculator import KRICalculator
from src.utils.logging_config import setup_logging


def main():
    """Run event-driven architecture demonstration."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Event-Driven Agent Architecture Demo")
    logger.info("=" * 60)
    
    # Create event bus
    event_bus = EventBus(enable_logging=True)
    logger.info("\n1. Created EventBus")
    
    # Create KRI calculator
    kri_calculator = KRICalculator()
    logger.info("2. Created KRI Calculator")
    
    # Create agents
    risk_manager = RiskManagerAgent(
        kri_calculator=kri_calculator,
        event_bus=event_bus,
        agent_id="RiskManager"
    )
    logger.info("3. Created Risk Manager Agent")
    
    market_analyst = MarketAnalystAgent(
        event_bus=event_bus,
        agent_id="MarketAnalyst",
        volatility_threshold=2.0,
        spread_threshold=0.5
    )
    logger.info("4. Created Market Analyst Agent")
    
    # Show event bus stats
    stats = event_bus.get_stats()
    logger.info(f"\nEvent Bus Stats: {stats}")
    
    # Simulate data updates
    logger.info("\n" + "=" * 60)
    logger.info("Simulating Data Updates")
    logger.info("=" * 60)
    
    # Create sample data with normal conditions
    logger.info("\n5. Publishing normal market data...")
    normal_data = pd.DataFrame({
        'unemployment': [4.0, 4.1, 4.0],
        'interest_rate': [2.5, 2.5, 2.6],
        'inflation': [2.0, 2.1, 2.0],
        'credit_spread': [1.5, 1.5, 1.6]
    })
    
    event1 = DataUpdateEvent(
        series_ids=['unemployment', 'interest_rate', 'inflation', 'credit_spread'],
        data=normal_data,
        timestamp=datetime.now(),
        source='FRED'
    )
    
    event_bus.publish('DataUpdateEvent', event1)
    
    # Create data with volatility spike
    logger.info("\n6. Publishing data with volatility spike...")
    volatile_data = pd.DataFrame({
        'unemployment': [4.0, 4.1, 5.5],  # Spike!
        'interest_rate': [2.5, 2.5, 2.6],
        'inflation': [2.0, 2.1, 3.5],  # Spike!
        'credit_spread': [1.5, 1.5, 1.6]
    })
    
    event2 = DataUpdateEvent(
        series_ids=['unemployment', 'interest_rate', 'inflation', 'credit_spread'],
        data=volatile_data,
        timestamp=datetime.now(),
        source='FRED'
    )
    
    event_bus.publish('DataUpdateEvent', event2)
    
    # Create data with credit spread widening
    logger.info("\n7. Publishing data with credit spread widening...")
    spread_data = pd.DataFrame({
        'unemployment': [5.5, 5.6, 5.7],
        'interest_rate': [2.6, 2.7, 2.8],
        'inflation': [3.5, 3.6, 3.7],
        'credit_spread': [1.6, 2.3, 2.8]  # Widening!
    })
    
    event3 = DataUpdateEvent(
        series_ids=['unemployment', 'interest_rate', 'inflation', 'credit_spread'],
        data=spread_data,
        timestamp=datetime.now(),
        source='FRED'
    )
    
    event_bus.publish('DataUpdateEvent', event3)
    
    # Get risk summary
    logger.info("\n" + "=" * 60)
    logger.info("Risk Assessment Summary")
    logger.info("=" * 60)
    
    risk_summary = risk_manager.get_current_risk_summary()
    
    logger.info(f"\nCurrent KRIs:")
    for kri_name, kri_value in risk_summary['kris'].items():
        alert_level = risk_summary['alerts'].get(kri_name, 'unknown')
        logger.info(f"  {kri_name}: {kri_value:.2f} (Risk Level: {alert_level})")
    
    logger.info(f"\nRisk Level Summary:")
    logger.info(f"  Critical Alerts: {risk_summary['n_critical']}")
    logger.info(f"  High Alerts: {risk_summary['n_high']}")
    logger.info(f"  Total Risk Signals: {risk_summary['n_risk_signals']}")
    
    # Show event history
    logger.info("\n" + "=" * 60)
    logger.info("Event History")
    logger.info("=" * 60)
    
    history = event_bus.get_event_history(limit=10)
    logger.info(f"\nTotal events published: {len(history)}")
    
    for i, event_record in enumerate(history, 1):
        logger.info(
            f"{i}. {event_record['event_type']} at {event_record['timestamp']} "
            f"({event_record['n_subscribers']} subscribers)"
        )
    
    # Final stats
    final_stats = event_bus.get_stats()
    logger.info(f"\nFinal Event Bus Stats:")
    logger.info(f"  Event Types: {final_stats['total_event_types']}")
    logger.info(f"  Total Subscribers: {final_stats['total_subscribers']}")
    logger.info(f"  Events Published: {final_stats['total_events_published']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
