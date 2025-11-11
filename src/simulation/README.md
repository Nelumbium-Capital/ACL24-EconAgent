# Simulation Module

This module implements Mesa-based agent-based modeling for stress testing and systemic risk analysis.

## Components

### 1. RiskSimulationModel (`model.py`)

The main simulation model that orchestrates interactions between banks and firms under various economic scenarios.

**Key Features:**
- Agent-based modeling using Mesa 3.x framework
- Tracks economic state variables (unemployment, GDP growth, interest rates, credit spreads)
- Collects comprehensive metrics on system liquidity, default rates, and network stress
- Supports custom economic scenarios

**Usage:**
```python
from simulation import RiskSimulationModel, RecessionScenario

# Create model with recession scenario
scenario = RecessionScenario(shock_start=12, shock_duration=18)
model = RiskSimulationModel(
    n_banks=10,
    n_firms=50,
    scenario=scenario,
    random_seed=42
)

# Run simulation
results = model.run_simulation(n_steps=100)

# Access results
print(f"Final default rate: {results['default_rate'].iloc[-1]}")
print(f"System liquidity: {results['system_liquidity'].iloc[-1]}")
```

### 2. Agents (`agents.py`)

#### BankAgent
Represents a financial institution with:
- Balance sheet (capital, deposits, loans, reserves)
- Risk management decisions
- Lending behavior based on capital ratios
- Liquidity management

**Key Metrics:**
- Capital ratio: capital / loans
- Liquidity ratio: reserves / deposits

#### FirmAgent
Represents a borrowing entity with:
- Borrowing needs
- Default probability (adjusted by economic conditions)
- Debt servicing behavior

### 3. Scenarios (`scenarios.py`)

Economic shock scenarios that define how economic variables evolve over time.

**Available Scenarios:**

#### BaselineScenario
Normal economic conditions with small random fluctuations.

#### RecessionScenario
Severe economic downturn with:
- Rising unemployment (up to 10%)
- Negative GDP growth (down to -3%)
- Widening credit spreads
- Gradual onset and recovery phases

```python
scenario = RecessionScenario(
    shock_start=12,      # Start recession at step 12
    shock_duration=18,   # Last for 18 steps
    peak_unemployment=0.10,
    min_gdp_growth=-0.03
)
```

#### InterestRateShockScenario
Sudden interest rate spike (e.g., aggressive monetary tightening):
- Immediate rate increase (e.g., 300 basis points)
- Secondary effects on unemployment and GDP
- Gradual normalization

```python
scenario = InterestRateShockScenario(
    shock_step=6,
    rate_increase=0.03,  # 300 basis points
    shock_duration=24
)
```

#### CreditCrisisScenario
Financial crisis with credit market disruption:
- Severe credit spread widening (up to 800 basis points)
- Acute phase followed by sustained crisis
- Gradual recovery

```python
scenario = CreditCrisisScenario(
    shock_start=10,
    shock_duration=20,
    peak_spread=0.08
)
```

#### CustomScenario
User-defined scenario with arbitrary paths for economic variables:

```python
# Define custom unemployment path
def custom_unemployment(step):
    if step < 10:
        return 0.04
    elif step < 30:
        return 0.04 + (step - 10) * 0.003  # Linear increase
    else:
        return 0.10

scenario = CustomScenario(
    name="Custom Scenario",
    unemployment_path=custom_unemployment,
    gdp_growth_path=lambda step: 0.02 - step * 0.001,
    description="Custom economic path"
)
```

### 4. Monte Carlo Engine (`monte_carlo.py`)

Runs multiple simulation instances to generate probability distributions of outcomes.

**Key Features:**
- Parallel execution support
- Random seed management for reproducibility
- Comprehensive statistical aggregation
- Risk metrics computation (VaR, CVaR, etc.)

**Usage:**
```python
from simulation import MonteCarloEngine, RecessionScenario

engine = MonteCarloEngine(
    n_banks=10,
    n_firms=50,
    n_steps=100
)

# Run Monte Carlo simulation
results = engine.run_monte_carlo(
    scenario=RecessionScenario(),
    n_simulations=100,
    random_seed=42,
    parallel=True
)

# Access results
summary_stats = results['summary_stats']
distributions = results['distributions']

# Compute risk metrics
risk_metrics = engine.compute_risk_metrics(distributions)
print(f"Mean default rate: {risk_metrics['default_rate']['mean']:.4f}")
print(f"VaR (95%): {risk_metrics['default_rate']['var_95']:.4f}")
```

**Convenience Function:**
```python
from simulation import run_stress_test, RecessionScenario

summary_stats, risk_metrics = run_stress_test(
    scenario=RecessionScenario(),
    n_simulations=100,
    n_banks=10,
    n_firms=50,
    n_steps=100,
    random_seed=42
)
```

**Scenario Comparison:**
```python
from simulation import MonteCarloEngine, SCENARIO_LIBRARY

engine = MonteCarloEngine(n_banks=10, n_firms=50, n_steps=100)

scenarios = [
    SCENARIO_LIBRARY['recession'],
    SCENARIO_LIBRARY['rate_shock'],
    SCENARIO_LIBRARY['credit_crisis']
]

comparison = engine.compare_scenarios(
    scenarios=scenarios,
    n_simulations=50,
    random_seed=42
)

# View comparison results
print(comparison[comparison['metric'] == 'default_rate'])
```

## Metrics Collected

The simulation collects the following metrics at each time step:

### Model-Level Metrics
- `system_liquidity`: Aggregate reserves / deposits across all banks
- `default_rate`: Proportion of firms in default
- `network_stress`: Proportion of banks below regulatory capital minimum
- `avg_capital_ratio`: Average capital ratio across banks
- `avg_liquidity_ratio`: Average liquidity ratio across banks
- `unemployment_rate`: Current unemployment rate
- `gdp_growth`: Current GDP growth rate
- `interest_rate`: Current interest rate
- `credit_spread`: Current credit spread

### Agent-Level Metrics
- `capital_ratio`: Bank's capital / loans
- `liquidity_ratio`: Bank's reserves / deposits
- `loan_quality`: Quality score of bank's loan portfolio (1.0 = perfect)
- `agent_type`: "bank" or "firm"

## Testing

Run the test scripts to verify functionality:

```bash
# Simple test
python scripts/test_simulation_simple.py

# Full test including Monte Carlo
python scripts/test_simulation.py
```

## Requirements

- mesa >= 3.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0

## Architecture Notes

This implementation uses Mesa 3.x API which differs from earlier versions:
- Agents are managed through `model.agents` AgentSet (not schedulers)
- Agents are created without explicit unique_id (auto-assigned)
- Model initialization uses `seed` parameter instead of separate random management
- Step execution is manual iteration through agents

## Future Enhancements

Potential extensions:
1. Interbank lending network
2. Contagion effects through counterparty exposures
3. Regulatory interventions (capital injections, liquidity support)
4. Market-based funding mechanisms
5. More sophisticated firm default models
6. Heterogeneous agent types (large/small banks, different sectors)
