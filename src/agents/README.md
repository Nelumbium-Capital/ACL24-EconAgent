# Event-Driven Agent Architecture

This module implements an event-driven architecture for risk management using specialized agents that communicate through a publish-subscribe event bus.

## Overview

The event-driven architecture enables modular, loosely-coupled components that react to system events in real-time. This design allows for:

- **Scalability**: Easy to add new agents without modifying existing code
- **Maintainability**: Clear separation of concerns between components
- **Testability**: Agents can be tested independently
- **Flexibility**: Dynamic subscription and event routing

## Components

### EventBus

The central message broker that enables publish-subscribe communication between agents.

**Key Features:**
- Thread-safe event publishing and subscription
- Event history tracking for debugging
- Comprehensive logging and statistics
- Support for multiple subscribers per event type

**Usage:**
```python
from src.agents import EventBus, DataUpdateEvent

# Create event bus
event_bus = EventBus(enable_logging=True)

# Subscribe to events
event_bus.subscribe('DataUpdateEvent', my_callback)

# Publish events
event = DataUpdateEvent(series_ids=['UNRATE'], data=df, timestamp=datetime.now())
event_bus.publish('DataUpdateEvent', event)
```

### Event Schemas

Strongly-typed event definitions for type safety and documentation.

**Available Events:**
- `DataUpdateEvent`: Published when new data is fetched
- `ForecastCompleteEvent`: Published when forecasting completes
- `SimulationCompleteEvent`: Published when stress testing completes
- `KRIUpdateEvent`: Published when KRIs are updated
- `RiskSignalEvent`: Published when risk signals are detected

### RiskManagerAgent

Monitors risk signals and computes risk assessments.

**Responsibilities:**
- Subscribe to data, forecast, and simulation events
- Compute KRIs using the KRI calculator
- Evaluate risk thresholds
- Publish KRI updates with alerts

**Event Flow:**
```
DataUpdateEvent → RiskManager → Compute KRIs → KRIUpdateEvent
ForecastCompleteEvent → RiskManager → Compute KRIs → KRIUpdateEvent
SimulationCompleteEvent → RiskManager → Compute KRIs → KRIUpdateEvent
RiskSignalEvent → RiskManager → Trigger Assessment → KRIUpdateEvent
```

**Usage:**
```python
from src.agents import RiskManagerAgent, EventBus
from src.kri.calculator import KRICalculator

event_bus = EventBus()
kri_calculator = KRICalculator()

risk_manager = RiskManagerAgent(
    kri_calculator=kri_calculator,
    event_bus=event_bus,
    agent_id="RiskManager"
)

# Agent automatically subscribes to events
# Get current risk summary
summary = risk_manager.get_current_risk_summary()
```

### MarketAnalystAgent

Analyzes market data and identifies risk signals.

**Responsibilities:**
- Monitor incoming data for anomalies
- Detect volatility spikes
- Detect credit spread widening
- Detect interest rate shocks
- Detect unemployment spikes
- Publish risk signals

**Detection Methods:**
- **Volatility Spike**: Recent volatility exceeds historical average by threshold
- **Credit Spread Widening**: Spread increases by more than threshold
- **Interest Rate Shock**: Rate changes by more than 0.5 percentage points
- **Unemployment Spike**: Unemployment increases by more than 0.5 percentage points

**Usage:**
```python
from src.agents import MarketAnalystAgent, EventBus

event_bus = EventBus()

market_analyst = MarketAnalystAgent(
    event_bus=event_bus,
    agent_id="MarketAnalyst",
    volatility_threshold=2.0,  # 2 standard deviations
    spread_threshold=0.5       # 0.5 percentage points
)

# Agent automatically subscribes to DataUpdateEvent
# Publishes RiskSignalEvent when anomalies detected
```

## Architecture Diagram

```
┌─────────────────┐
│  Data Pipeline  │
└────────┬────────┘
         │ DataUpdateEvent
         ▼
┌─────────────────┐
│   Event Bus     │◄──────────────────┐
└────────┬────────┘                   │
         │                            │
         ├──────────────┬─────────────┤
         │              │             │
         ▼              ▼             │
┌─────────────┐  ┌──────────────┐    │
│   Market    │  │     Risk     │    │
│  Analyst    │  │   Manager    │    │
└──────┬──────┘  └──────┬───────┘    │
       │                │             │
       │ RiskSignalEvent│ KRIUpdateEvent
       └────────────────┴─────────────┘
```

## Example: Complete Event Flow

```python
import pandas as pd
from datetime import datetime
from src.agents import (
    EventBus,
    RiskManagerAgent,
    MarketAnalystAgent,
    DataUpdateEvent
)
from src.kri.calculator import KRICalculator

# 1. Setup
event_bus = EventBus(enable_logging=True)
kri_calculator = KRICalculator()

# 2. Create agents
risk_manager = RiskManagerAgent(kri_calculator, event_bus)
market_analyst = MarketAnalystAgent(event_bus)

# 3. Publish data update
data = pd.DataFrame({
    'unemployment': [4.0, 4.5, 5.5],  # Spike!
    'interest_rate': [2.5, 2.6, 2.7]
})

event = DataUpdateEvent(
    series_ids=['unemployment', 'interest_rate'],
    data=data,
    timestamp=datetime.now()
)

event_bus.publish('DataUpdateEvent', event)

# 4. Agents automatically process:
#    - MarketAnalyst detects unemployment spike
#    - MarketAnalyst publishes RiskSignalEvent
#    - RiskManager receives RiskSignalEvent
#    - RiskManager computes KRIs
#    - RiskManager publishes KRIUpdateEvent

# 5. Get results
summary = risk_manager.get_current_risk_summary()
print(f"KRIs: {summary['kris']}")
print(f"Alerts: {summary['alerts']}")
```

## Testing

Run the demo script to see the event-driven architecture in action:

```bash
python examples/event_driven_demo.py
```

This demonstrates:
- Event bus creation and subscription
- Agent initialization
- Data updates with normal and stressed conditions
- Risk signal detection
- KRI computation and alerts
- Event history tracking

## Extension Points

### Adding New Agents

1. Create a new agent class
2. Subscribe to relevant events in `__init__`
3. Implement event handler methods
4. Publish new events as needed

Example:
```python
class CustomAgent:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.event_bus.subscribe('DataUpdateEvent', self.on_data_update)
    
    def on_data_update(self, event: DataUpdateEvent):
        # Process event
        # Publish results
        pass
```

### Adding New Event Types

1. Define event dataclass in `events.py`
2. Update agents to publish/subscribe to new event
3. Document event schema and usage

## Best Practices

1. **Event Naming**: Use descriptive names ending in "Event" (e.g., `DataUpdateEvent`)
2. **Error Handling**: Wrap event handlers in try-except to prevent cascading failures
3. **Logging**: Use appropriate log levels (INFO for normal flow, WARNING for signals, ERROR for failures)
4. **Thread Safety**: EventBus is thread-safe, but agent state should be protected if accessed from multiple threads
5. **Testing**: Test agents independently by mocking the event bus

## Performance Considerations

- Event history is limited to prevent memory growth
- Subscribers are called synchronously - avoid long-running operations in handlers
- Consider using async/await for I/O-bound operations
- Monitor event bus statistics to detect bottlenecks

## Troubleshooting

**Events not being received:**
- Check subscription is called before publishing
- Verify event type string matches exactly
- Check logs for subscription confirmation

**Agent not responding:**
- Check for exceptions in event handlers (logged as errors)
- Verify agent is subscribed to correct event types
- Check event bus statistics to confirm events are published

**Memory growth:**
- Event history is automatically limited
- Call `event_bus.clear_history()` periodically if needed
- Monitor agent state for unbounded growth
