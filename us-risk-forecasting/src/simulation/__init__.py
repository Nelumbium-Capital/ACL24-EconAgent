"""
Simulation module for agent-based stress testing.
"""
from .model import RiskSimulationModel
from .agents import BankAgent, FirmAgent
from .scenarios import (
    EconomicScenario,
    BaselineScenario,
    RecessionScenario,
    InterestRateShockScenario,
    CreditCrisisScenario,
    CustomScenario,
    get_scenario,
    SCENARIO_LIBRARY
)
from .monte_carlo import MonteCarloEngine, run_stress_test

__all__ = [
    'RiskSimulationModel',
    'BankAgent',
    'FirmAgent',
    'EconomicScenario',
    'BaselineScenario',
    'RecessionScenario',
    'InterestRateShockScenario',
    'CreditCrisisScenario',
    'CustomScenario',
    'get_scenario',
    'SCENARIO_LIBRARY',
    'MonteCarloEngine',
    'run_stress_test'
]
