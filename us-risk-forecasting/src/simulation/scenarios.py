"""
Economic scenario generators for stress testing.
"""
from mesa import Model
import numpy as np
from typing import Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class EconomicScenario:
    """
    Base class for economic shock scenarios.
    
    Scenarios define how economic variables (unemployment, GDP growth, interest rates,
    credit spreads) evolve over the simulation period.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize an economic scenario.
        
        Args:
            name: Name of the scenario
            description: Description of the scenario
        """
        self.name = name
        self.description = description
        logger.info(f"Initialized scenario: {name}")
    
    def apply_shock(self, model: Model, step: int):
        """
        Apply economic shock at given simulation step.
        
        Args:
            model: The simulation model
            step: Current simulation step
        """
        raise NotImplementedError("Subclasses must implement apply_shock()")
    
    def reset(self, model: Model):
        """
        Reset economic variables to baseline values.
        
        Args:
            model: The simulation model
        """
        model.unemployment_rate = 0.04
        model.gdp_growth = 0.02
        model.interest_rate = 0.03
        model.credit_spread = 0.02


class BaselineScenario(EconomicScenario):
    """
    Baseline scenario with no shocks - normal economic conditions.
    """
    
    def __init__(self):
        super().__init__(
            name="Baseline",
            description="Normal economic conditions with no shocks"
        )
    
    def apply_shock(self, model: Model, step: int):
        """No shocks applied in baseline scenario."""
        # Add small random fluctuations for realism
        model.unemployment_rate = 0.04 + np.random.normal(0, 0.001)
        model.gdp_growth = 0.02 + np.random.normal(0, 0.002)
        model.interest_rate = 0.03 + np.random.normal(0, 0.001)
        model.credit_spread = 0.02 + np.random.normal(0, 0.001)


class RecessionScenario(EconomicScenario):
    """
    Recession scenario with unemployment spike and GDP contraction.
    
    Models a severe economic downturn with rising unemployment,
    negative GDP growth, and widening credit spreads.
    """
    
    def __init__(
        self,
        shock_start: int = 12,
        shock_duration: int = 18,
        peak_unemployment: float = 0.10,
        min_gdp_growth: float = -0.03
    ):
        """
        Initialize recession scenario.
        
        Args:
            shock_start: Step at which recession begins
            shock_duration: Duration of recession in steps
            peak_unemployment: Peak unemployment rate during recession
            min_gdp_growth: Minimum (most negative) GDP growth rate
        """
        super().__init__(
            name="Recession",
            description=f"Severe recession starting at step {shock_start}"
        )
        self.shock_start = shock_start
        self.shock_duration = shock_duration
        self.peak_unemployment = peak_unemployment
        self.min_gdp_growth = min_gdp_growth
    
    def apply_shock(self, model: Model, step: int):
        """Apply recession shock with gradual onset and recovery."""
        if step < self.shock_start:
            # Pre-shock: normal conditions
            model.unemployment_rate = 0.04
            model.gdp_growth = 0.02
            model.credit_spread = 0.02
            
        elif step < self.shock_start + self.shock_duration // 2:
            # Onset phase: conditions deteriorate
            progress = (step - self.shock_start) / (self.shock_duration // 2)
            
            model.unemployment_rate = 0.04 + (self.peak_unemployment - 0.04) * progress
            model.gdp_growth = 0.02 + (self.min_gdp_growth - 0.02) * progress
            model.credit_spread = 0.02 + 0.04 * progress
            
            logger.debug(f"Recession onset: step {step}, unemployment={model.unemployment_rate:.4f}")
            
        elif step < self.shock_start + self.shock_duration:
            # Recovery phase: conditions improve
            progress = (step - self.shock_start - self.shock_duration // 2) / (self.shock_duration // 2)
            
            model.unemployment_rate = self.peak_unemployment - (self.peak_unemployment - 0.05) * progress
            model.gdp_growth = self.min_gdp_growth + (0.02 - self.min_gdp_growth) * progress
            model.credit_spread = 0.06 - 0.03 * progress
            
            logger.debug(f"Recession recovery: step {step}, unemployment={model.unemployment_rate:.4f}")
            
        else:
            # Post-shock: return to normal
            model.unemployment_rate = 0.05  # Slightly elevated
            model.gdp_growth = 0.02
            model.credit_spread = 0.025


class InterestRateShockScenario(EconomicScenario):
    """
    Sudden interest rate spike scenario.
    
    Models a rapid increase in interest rates, such as aggressive
    monetary policy tightening to combat inflation.
    """
    
    def __init__(
        self,
        shock_step: int = 6,
        rate_increase: float = 0.03,
        shock_duration: int = 24
    ):
        """
        Initialize interest rate shock scenario.
        
        Args:
            shock_step: Step at which rate shock occurs
            rate_increase: Magnitude of rate increase (e.g., 0.03 = 300 basis points)
            shock_duration: How long elevated rates persist
        """
        super().__init__(
            name="Interest Rate Shock",
            description=f"Sudden {rate_increase*100:.0f}bp rate increase at step {shock_step}"
        )
        self.shock_step = shock_step
        self.rate_increase = rate_increase
        self.shock_duration = shock_duration
    
    def apply_shock(self, model: Model, step: int):
        """Apply sudden interest rate increase."""
        if step < self.shock_step:
            # Pre-shock: normal conditions
            model.interest_rate = 0.03
            model.unemployment_rate = 0.04
            model.gdp_growth = 0.02
            model.credit_spread = 0.02
            
        elif step == self.shock_step:
            # Shock occurs: sudden rate increase
            model.interest_rate = 0.03 + self.rate_increase
            logger.info(f"Interest rate shock: rate increased to {model.interest_rate:.4f}")
            
        elif step < self.shock_step + self.shock_duration:
            # During shock: rates remain elevated, economy slows
            model.interest_rate = 0.03 + self.rate_increase
            
            # Secondary effects on economy
            time_since_shock = step - self.shock_step
            impact_factor = min(time_since_shock / 12, 1.0)  # Effects build over 12 steps
            
            model.unemployment_rate = 0.04 + 0.02 * impact_factor
            model.gdp_growth = 0.02 - 0.03 * impact_factor
            model.credit_spread = 0.02 + 0.02 * impact_factor
            
        else:
            # Post-shock: gradual normalization
            steps_since_end = step - (self.shock_step + self.shock_duration)
            normalization_progress = min(steps_since_end / 12, 1.0)
            
            model.interest_rate = (0.03 + self.rate_increase) - self.rate_increase * normalization_progress
            model.unemployment_rate = 0.06 - 0.01 * normalization_progress
            model.gdp_growth = -0.01 + 0.03 * normalization_progress
            model.credit_spread = 0.04 - 0.015 * normalization_progress


class CreditCrisisScenario(EconomicScenario):
    """
    Credit crisis scenario with credit spread widening.
    
    Models a financial crisis with severe credit market disruption,
    widening spreads, and reduced lending.
    """
    
    def __init__(
        self,
        shock_start: int = 10,
        shock_duration: int = 20,
        peak_spread: float = 0.08
    ):
        """
        Initialize credit crisis scenario.
        
        Args:
            shock_start: Step at which crisis begins
            shock_duration: Duration of crisis
            peak_spread: Peak credit spread during crisis
        """
        super().__init__(
            name="Credit Crisis",
            description=f"Credit market disruption starting at step {shock_start}"
        )
        self.shock_start = shock_start
        self.shock_duration = shock_duration
        self.peak_spread = peak_spread
    
    def apply_shock(self, model: Model, step: int):
        """Apply credit crisis with spread widening."""
        if step < self.shock_start:
            # Pre-crisis: normal conditions
            model.credit_spread = 0.02
            model.unemployment_rate = 0.04
            model.gdp_growth = 0.02
            model.interest_rate = 0.03
            
        elif step < self.shock_start + 5:
            # Acute phase: rapid spread widening
            progress = (step - self.shock_start) / 5
            
            model.credit_spread = 0.02 + (self.peak_spread - 0.02) * progress
            model.unemployment_rate = 0.04 + 0.03 * progress
            model.gdp_growth = 0.02 - 0.04 * progress
            
            logger.warning(f"Credit crisis acute phase: spread={model.credit_spread:.4f}")
            
        elif step < self.shock_start + self.shock_duration:
            # Sustained crisis: spreads remain elevated
            model.credit_spread = self.peak_spread
            model.unemployment_rate = 0.07
            model.gdp_growth = -0.02
            
        else:
            # Recovery: gradual normalization
            steps_since_end = step - (self.shock_start + self.shock_duration)
            recovery_progress = min(steps_since_end / 15, 1.0)
            
            model.credit_spread = self.peak_spread - (self.peak_spread - 0.025) * recovery_progress
            model.unemployment_rate = 0.07 - 0.02 * recovery_progress
            model.gdp_growth = -0.02 + 0.04 * recovery_progress


class CustomScenario(EconomicScenario):
    """
    Custom scenario with user-defined shock paths.
    
    Allows users to specify arbitrary paths for economic variables.
    """
    
    def __init__(
        self,
        name: str,
        unemployment_path: Optional[Callable[[int], float]] = None,
        gdp_growth_path: Optional[Callable[[int], float]] = None,
        interest_rate_path: Optional[Callable[[int], float]] = None,
        credit_spread_path: Optional[Callable[[int], float]] = None,
        description: str = "Custom user-defined scenario"
    ):
        """
        Initialize custom scenario.
        
        Args:
            name: Name of the scenario
            unemployment_path: Function mapping step -> unemployment rate
            gdp_growth_path: Function mapping step -> GDP growth rate
            interest_rate_path: Function mapping step -> interest rate
            credit_spread_path: Function mapping step -> credit spread
            description: Description of the scenario
        """
        super().__init__(name=name, description=description)
        
        self.unemployment_path = unemployment_path or (lambda step: 0.04)
        self.gdp_growth_path = gdp_growth_path or (lambda step: 0.02)
        self.interest_rate_path = interest_rate_path or (lambda step: 0.03)
        self.credit_spread_path = credit_spread_path or (lambda step: 0.02)
    
    def apply_shock(self, model: Model, step: int):
        """Apply custom shock paths."""
        model.unemployment_rate = self.unemployment_path(step)
        model.gdp_growth = self.gdp_growth_path(step)
        model.interest_rate = self.interest_rate_path(step)
        model.credit_spread = self.credit_spread_path(step)
        
        logger.debug(f"Custom scenario step {step}: U={model.unemployment_rate:.4f}, "
                    f"GDP={model.gdp_growth:.4f}, R={model.interest_rate:.4f}")


# Predefined scenario library
SCENARIO_LIBRARY: Dict[str, EconomicScenario] = {
    'baseline': BaselineScenario(),
    'recession': RecessionScenario(),
    'rate_shock': InterestRateShockScenario(),
    'credit_crisis': CreditCrisisScenario()
}


def get_scenario(scenario_name: str) -> EconomicScenario:
    """
    Get a scenario from the library by name.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        EconomicScenario instance
        
    Raises:
        ValueError: If scenario name not found
    """
    if scenario_name not in SCENARIO_LIBRARY:
        raise ValueError(f"Unknown scenario: {scenario_name}. "
                        f"Available: {list(SCENARIO_LIBRARY.keys())}")
    
    return SCENARIO_LIBRARY[scenario_name]
