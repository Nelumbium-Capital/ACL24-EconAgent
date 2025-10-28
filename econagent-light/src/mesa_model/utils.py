"""
Economic utility functions for EconAgent-Light Mesa model.
Implements original economic calculations from ACL24-EconAgent.
"""

import numpy as np
from typing import List, Tuple, Dict, Any

# Original tax brackets from simulate_utils.py (monthly, scaled)
US_TAX_BRACKETS = [0, 97, 394.75, 842, 1607.25, 2041, 5103]  # Monthly brackets
US_TAX_BRACKETS_SCALED = [b * 100 / 12 for b in US_TAX_BRACKETS]

# 2018 U.S. Federal tax rates (approximated for simulation)
US_TAX_RATES_2018 = [0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

def pareto_skill_distribution(
    n_agents: int, 
    pareto_param: float = 8.0, 
    max_skill_multiplier: float = 950.0
) -> np.ndarray:
    """
    Generate Pareto-distributed skills for agents.
    
    Args:
        n_agents: Number of agents
        pareto_param: Pareto distribution parameter (from original config)
        max_skill_multiplier: Maximum skill multiplier
        
    Returns:
        Array of skill values for agents
    """
    # Generate Pareto distribution
    skills = np.random.pareto(pareto_param, n_agents) + 1.0
    
    # Scale and clip to reasonable range
    skills = skills / np.max(skills) * max_skill_multiplier
    skills = np.clip(skills, 1.0, max_skill_multiplier)
    
    return skills

def compute_income_tax(
    income: float, 
    tax_brackets: List[float], 
    tax_rates: List[float]
) -> float:
    """
    Compute progressive income tax using bracket system.
    
    Args:
        income: Gross income to tax
        tax_brackets: Tax bracket thresholds
        tax_rates: Tax rates for each bracket
        
    Returns:
        Total tax owed
    """
    if income <= 0 or not tax_brackets or not tax_rates:
        return 0.0
    
    tax_owed = 0.0
    remaining_income = income
    
    for i in range(1, len(tax_brackets)):
        if remaining_income <= 0:
            break
        
        # Calculate bracket size
        bracket_size = tax_brackets[i] - tax_brackets[i-1]
        
        # Amount taxable in this bracket
        taxable_in_bracket = min(remaining_income, bracket_size)
        
        # Apply tax rate for this bracket
        if i-1 < len(tax_rates):
            tax_owed += taxable_in_bracket * tax_rates[i-1]
        
        remaining_income -= taxable_in_bracket
    
    # Handle income above highest bracket
    if remaining_income > 0 and len(tax_rates) > 0:
        tax_owed += remaining_income * tax_rates[-1]
    
    return max(0.0, tax_owed)

def update_wages_and_prices(
    current_wage: float,
    current_price: float,
    employment_rate: float,
    inventory_level: float,
    max_wage_inflation: float = 0.05,
    max_price_inflation: float = 0.1,
    wage_adjustment_rate: float = 0.05,  # alpha_w
    price_adjustment_rate: float = 0.10,  # alpha_P
    target_employment: float = 0.95,
    target_inventory: float = 1000.0
) -> Tuple[float, float]:
    """
    Update wages and prices based on market conditions.
    Implements original equations (7) and (8) from the paper.
    
    Args:
        current_wage: Current average wage level
        current_price: Current price level
        employment_rate: Current employment rate [0,1]
        inventory_level: Current goods inventory
        max_wage_inflation: Maximum wage inflation rate
        max_price_inflation: Maximum price inflation rate
        wage_adjustment_rate: Wage adjustment sensitivity (alpha_w)
        price_adjustment_rate: Price adjustment sensitivity (alpha_P)
        target_employment: Target employment rate
        target_inventory: Target inventory level
        
    Returns:
        Tuple of (new_wage, new_price)
    """
    # Wage adjustment based on employment (equation 7 from paper)
    employment_gap = employment_rate - target_employment
    wage_change_rate = wage_adjustment_rate * employment_gap
    wage_change_rate = np.clip(wage_change_rate, -max_wage_inflation, max_wage_inflation)
    
    new_wage = current_wage * (1.0 + wage_change_rate)
    new_wage = max(0.1, new_wage)  # Minimum wage floor
    
    # Price adjustment based on inventory/demand (equation 8 from paper)
    # Low inventory -> higher prices, high inventory -> lower prices
    inventory_ratio = inventory_level / target_inventory if target_inventory > 0 else 1.0
    inventory_gap = 1.0 - inventory_ratio  # Positive when inventory is low
    
    price_change_rate = price_adjustment_rate * inventory_gap
    price_change_rate = np.clip(price_change_rate, -max_price_inflation, max_price_inflation)
    
    new_price = current_price * (1.0 + price_change_rate)
    new_price = max(0.1, new_price)  # Minimum price floor
    
    return new_wage, new_price

def taylor_rule_interest_rate(
    current_rate: float,
    inflation_rate: float,
    unemployment_rate: float,
    natural_rate: float = 0.01,
    inflation_target: float = 0.02,
    unemployment_target: float = 0.04,
    alpha_pi: float = 0.5,
    alpha_u: float = 0.5
) -> float:
    """
    Update interest rate using Taylor rule (equation 12 from paper).
    
    Args:
        current_rate: Current interest rate
        inflation_rate: Current inflation rate
        unemployment_rate: Current unemployment rate
        natural_rate: Natural interest rate (rn)
        inflation_target: Target inflation rate (pi_t)
        unemployment_target: Target unemployment rate (un)
        alpha_pi: Inflation response parameter
        alpha_u: Unemployment response parameter
        
    Returns:
        New interest rate
    """
    # Taylor rule: r = rn + alpha_pi * (pi - pi_t) - alpha_u * (u - un)
    inflation_gap = inflation_rate - inflation_target
    unemployment_gap = unemployment_rate - unemployment_target
    
    new_rate = (natural_rate + 
                alpha_pi * inflation_gap - 
                alpha_u * unemployment_gap)
    
    # Constrain interest rate to reasonable bounds
    new_rate = np.clip(new_rate, 0.0, 0.20)  # 0% to 20%
    
    return new_rate

def format_currency(amount: float) -> str:
    """Format currency for display."""
    return f"${amount:,.2f}"

def format_percentage(rate: float) -> str:
    """Format percentage for display."""
    return f"{rate:.1%}"

def calculate_utility(
    consumption: float,
    labor: float,
    wealth: float,
    utility_type: str = "isoelastic_coin_minus_labor",
    isoelastic_eta: float = 0.5,
    labor_cost: float = 1.0,
    labor_exponent: float = 2.0
) -> float:
    """
    Calculate agent utility based on consumption, labor, and wealth.
    Implements utility functions from original agent_reward_type.
    
    Args:
        consumption: Consumption amount
        labor: Labor amount (0-1)
        wealth: Current wealth
        utility_type: Type of utility function
        isoelastic_eta: Isoelastic utility parameter
        labor_cost: Cost coefficient for labor
        labor_exponent: Exponent for labor cost
        
    Returns:
        Utility value
    """
    if utility_type == "isoelastic_coin_minus_labor":
        # Isoelastic utility with labor cost
        if wealth <= 0:
            wealth_utility = -np.inf
        else:
            wealth_utility = (wealth ** (1 - isoelastic_eta)) / (1 - isoelastic_eta)
        
        labor_cost_term = labor_cost * (labor ** labor_exponent)
        
        return wealth_utility - labor_cost_term
    
    elif utility_type == "coin_minus_labor_cost":
        # Simple linear utility with labor cost
        labor_cost_term = labor_cost * (labor ** labor_exponent)
        return wealth - labor_cost_term
    
    else:
        # Default: simple wealth utility
        return wealth

def validate_economic_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize economic parameters.
    
    Args:
        params: Dictionary of economic parameters
        
    Returns:
        Validated parameters dictionary
    """
    validated = params.copy()
    
    # Ensure positive values for key parameters
    positive_params = [
        'productivity', 'pareto_param', 'payment_max_skill_multiplier',
        'labor_hours', 'base_interest_rate'
    ]
    
    for param in positive_params:
        if param in validated:
            validated[param] = max(0.001, float(validated[param]))
    
    # Ensure rates are in [0,1] range
    rate_params = [
        'skill_change', 'price_change', 'max_price_inflation', 
        'max_wage_inflation', 'consumption_rate_step'
    ]
    
    for param in rate_params:
        if param in validated:
            validated[param] = np.clip(float(validated[param]), 0.0, 1.0)
    
    # Ensure integer parameters
    int_params = ['n_agents', 'episode_length', 'labor_hours']
    
    for param in int_params:
        if param in validated:
            validated[param] = max(1, int(validated[param]))
    
    return validated

def calculate_market_indicators(
    agents: List[Any],
    current_price: float,
    current_wage: float,
    inventory: float
) -> Dict[str, float]:
    """
    Calculate various market indicators for analysis.
    
    Args:
        agents: List of economic agents
        current_price: Current goods price
        current_wage: Current average wage
        inventory: Current goods inventory
        
    Returns:
        Dictionary of market indicators
    """
    if not agents:
        return {}
    
    # Employment metrics
    employed_agents = [a for a in agents if getattr(a, 'worked_this_month', False)]
    employment_rate = len(employed_agents) / len(agents)
    
    # Wealth distribution
    wealths = [getattr(a, 'wealth', 0) for a in agents]
    avg_wealth = np.mean(wealths)
    median_wealth = np.median(wealths)
    wealth_std = np.std(wealths)
    
    # Income distribution
    incomes = [getattr(a, 'last_income', 0) for a in agents]
    avg_income = np.mean(incomes)
    
    # Consumption metrics
    consumptions = [getattr(a, 'actual_consumption', 0) for a in agents]
    total_consumption = sum(consumptions)
    avg_consumption = np.mean(consumptions)
    
    return {
        'employment_rate': employment_rate,
        'unemployment_rate': 1.0 - employment_rate,
        'avg_wealth': avg_wealth,
        'median_wealth': median_wealth,
        'wealth_inequality': wealth_std / avg_wealth if avg_wealth > 0 else 0,
        'avg_income': avg_income,
        'total_consumption': total_consumption,
        'avg_consumption': avg_consumption,
        'price_level': current_price,
        'wage_level': current_wage,
        'inventory_level': inventory,
        'consumption_rate': total_consumption / (avg_wealth * len(agents)) if avg_wealth > 0 else 0
    }