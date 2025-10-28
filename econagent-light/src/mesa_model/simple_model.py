"""
Simplified Mesa model for web UI that works without LightAgent dependencies.
Focuses on core economic simulation with fallback decision-making.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from collections import deque

import mesa
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from .utils import (
    pareto_skill_distribution,
    compute_income_tax,
    update_wages_and_prices,
    taylor_rule_interest_rate,
    US_TAX_BRACKETS,
    US_TAX_RATES_2018
)

logger = logging.getLogger(__name__)

class SimpleEconAgent(Agent):
    """Simplified economic agent with fallback decision-making."""
    
    def __init__(self, unique_id: int, model, skill: float, initial_wealth: float = 100.0):
        super().__init__(unique_id, model)
        
        # Agent attributes
        self.skill = skill
        self.wealth = initial_wealth
        self.age = model.random.randint(18, 65)
        self.job = "Unemployment" if model.random.random() < 0.1 else "Employed"
        
        # Economic state
        self.monthly_wage = skill * model.average_wage * model.labor_hours
        self.last_income = 0.0
        self.last_work_decision = 0.5
        self.last_consumption_decision = 0.3
        self.worked_this_month = False
        self.consumption_spending = 0.0
        self.actual_consumption = 0.0
        
        # Financial tracking
        self.tax_paid = 0.0
        self.redistribution_received = 0.0
        self.interest_earned = 0.0
        
        # Decision history
        self.decision_history = deque(maxlen=12)
    
    def step(self):
        """Make economic decisions using simple heuristics."""
        # Update wage based on market conditions
        self.monthly_wage = self.skill * self.model.average_wage * self.model.labor_hours
        
        # Use original complex_actions logic from the paper
        price = self.model.goods_price
        wealth = self.wealth
        max_income = self.monthly_wage
        last_income = self.last_income
        interest_rate = self.model.interest_rate
        
        # Original work function: work_income_wealth with gamma=0.1
        gamma = 0.1
        work_probability = (max_income / (wealth * (1 + interest_rate) + 1e-8)) ** gamma
        work_decision = int(self.model.random.random() < work_probability)
        
        self.worked_this_month = bool(work_decision)
        self.last_work_decision = work_probability
        
        # Calculate income
        if self.worked_this_month:
            self.last_income = self.monthly_wage
            self.wealth += self.last_income
            if self.job == "Unemployment":
                self.job = "Employed"
        else:
            self.last_income = 0.0
            # Might become unemployed if didn't work
            if self.job == "Employed" and self.model.random.random() < 0.05:
                self.job = "Unemployment"
        
        # Use original consumption functions from complex_actions
        curr_income = work_decision * max_income
        beta = 0.1
        h = 1
        
        # Choose consumption function (original has two: consumption_len and consumption_cats)
        if not hasattr(self, 'consumption_fun_idx'):
            self.consumption_fun_idx = self.model.random.choice([0, 1])
        
        if self.consumption_fun_idx == 0:
            # consumption_len function
            c = (price / (1e-8 + wealth + curr_income)) ** beta
            c = min(max(c // 0.02, 0), 50)
        else:
            # consumption_cats function  
            h1 = h / (1 + interest_rate)
            g = curr_income / (last_income + 1e-8) - 1
            d = wealth / (last_income + 1e-8) - h1
            c = 1 + (d - h1 * g) / (1 + g + 1e-8)
            c = min(max(c * curr_income / (wealth + curr_income + 1e-8) // 0.02, 0), 50)
        
        consumption_propensity = c * 0.02  # Convert back to [0,1] range
        self.last_consumption_decision = consumption_propensity
        
        # Calculate consumption spending
        available_funds = self.wealth
        self.consumption_spending = available_funds * consumption_propensity
        
        # Store decision history
        self.decision_history.append({
            'step': self.model.current_step,
            'work_decision': work_propensity,
            'consumption_decision': consumption_propensity,
            'worked': self.worked_this_month,
            'income': self.last_income,
            'wealth': self.wealth,
            'job': self.job
        })

class SimpleEconModel(Model):
    """Simplified economic model for web UI."""
    
    def __init__(
        self,
        n_agents: int = 100,
        episode_length: int = 240,
        random_seed: Optional[int] = None,
        productivity: float = 1.0,
        skill_change: float = 0.02,
        price_change: float = 0.02,
        max_price_inflation: float = 0.1,
        max_wage_inflation: float = 0.05,
        pareto_param: float = 8.0,
        payment_max_skill_multiplier: float = 950.0,
        labor_hours: int = 168,
        base_interest_rate: float = 0.01,
        log_frequency: int = 3
    ):
        super().__init__()
        
        # Set random seed
        if random_seed is not None:
            self.random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Model parameters
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.current_step = 0
        self.log_frequency = log_frequency
        
        # Economic parameters
        self.productivity = productivity
        self.skill_change = skill_change
        self.price_change = price_change
        self.max_price_inflation = max_price_inflation
        self.max_wage_inflation = max_wage_inflation
        self.pareto_param = pareto_param
        self.payment_max_skill_multiplier = payment_max_skill_multiplier
        self.labor_hours = labor_hours
        self.base_interest_rate = base_interest_rate
        
        # Economic state
        self.goods_inventory = 0.0
        self.goods_price = 1.0
        self.average_wage = 1.0
        self.interest_rate = base_interest_rate
        self.inflation_rate = 0.0
        self.unemployment_rate = 0.0
        
        # Economic history
        self.price_history = [self.goods_price]
        self.wage_history = [self.average_wage]
        self.interest_rate_history = [self.interest_rate]
        
        # Tax system
        self.tax_brackets = US_TAX_BRACKETS.copy()
        self.tax_rates = US_TAX_RATES_2018.copy()
        self.government_revenue = 0.0
        self.redistribution_pool = 0.0
        
        # Create scheduler and agents
        self.schedule = RandomActivation(self)
        
        self._create_agents()
        
        # Data collection
        self.model_data = []
        self.agent_data = []
        
        # Collect initial data
        self._collect_data()
        
        logger.info(f"SimpleEconModel initialized: {n_agents} agents, {episode_length} steps")
    
    def _create_agents(self):
        """Create economic agents."""
        skills = pareto_skill_distribution(
            self.n_agents,
            self.pareto_param,
            self.payment_max_skill_multiplier
        )
        
        for i in range(self.n_agents):
            agent = SimpleEconAgent(
                unique_id=i,
                model=self,
                skill=skills[i],
                initial_wealth=self.random.uniform(100, 1000)
            )
            
            self.schedule.add(agent)
    
    def step(self):
        """Execute one simulation step."""
        self.current_step += 1
        
        # Agent decision phase
        self.schedule.step()
        
        # Production phase
        self._production_phase()
        
        # Consumption phase
        self._consumption_phase()
        
        # Economic updates
        self._update_wages_and_prices()
        self._update_interest_rates()
        
        # Tax and redistribution
        self._tax_and_redistribution_phase()
        
        # Data collection
        self._collect_data()
        
        # Logging
        if self.current_step % self.log_frequency == 0:
            self._log_progress()
        
        # Check completion
        if self.current_step >= self.episode_length:
            self.running = False
    
    def _production_phase(self):
        """Calculate production using original ai-economist logic."""
        total_production = 0.0
        
        agents = self.schedule.agents
        for agent in agents:
            if agent.worked_this_month:
                # Original production: labor_hours * skill * productivity
                production = self.labor_hours * agent.skill * self.productivity
                total_production += production
        
        # Add to total products (inventory)
        self.goods_inventory += total_production
    
    def _consumption_phase(self):
        """Process consumption."""
        agents = self.schedule.agents
        total_consumption_demand = sum(agent.consumption_spending for agent in agents)
        
        # Limit consumption by inventory
        actual_consumption = min(total_consumption_demand, self.goods_inventory)
        consumption_ratio = actual_consumption / total_consumption_demand if total_consumption_demand > 0 else 1.0
        
        # Allocate consumption
        for agent in agents:
            if agent.consumption_spending > 0:
                agent.actual_consumption = agent.consumption_spending * consumption_ratio
                agent.wealth -= agent.actual_consumption
            else:
                agent.actual_consumption = 0.0
        
        self.goods_inventory -= actual_consumption
        self.goods_inventory = max(0.0, self.goods_inventory)
    
    def _update_wages_and_prices(self):
        """Update wages and prices using original ai-economist logic."""
        agents = self.schedule.agents
        
        # Calculate employment metrics
        employed_agents = [a for a in agents if a.worked_this_month]
        employment_rate = len(employed_agents) / self.n_agents
        self.unemployment_rate = 1.0 - employment_rate
        
        # Original price update logic from SimpleConsumption component
        # Price increases when demand > supply (low inventory)
        total_demand = sum(getattr(a, 'actual_consumption', 0) for a in agents)
        supply_demand_ratio = self.goods_inventory / (total_demand + 1e-8)
        
        # Price adjustment based on supply/demand
        if supply_demand_ratio < 1.0:  # Demand > Supply
            price_change = min(self.max_price_inflation, 0.1 * (1.0 - supply_demand_ratio))
        else:  # Supply >= Demand
            price_change = max(-self.max_price_inflation, -0.05 * (supply_demand_ratio - 1.0))
        
        new_price = self.goods_price * (1.0 + price_change)
        new_price = max(0.1, new_price)
        
        # Wage adjustment based on employment (original logic)
        employment_gap = employment_rate - 0.95  # Target 95% employment
        wage_change = min(max(0.05 * employment_gap, -self.max_wage_inflation), self.max_wage_inflation)
        
        new_wage = self.average_wage * (1.0 + wage_change)
        new_wage = max(0.1, new_wage)
        
        # Calculate inflation
        if len(self.price_history) > 0:
            self.inflation_rate = (new_price - self.price_history[-1]) / self.price_history[-1]
        else:
            self.inflation_rate = 0.0
        
        # Update values
        self.average_wage = new_wage
        self.goods_price = new_price
        self.wage_history.append(new_wage)
        self.price_history.append(new_price)
        
        # Update agent wages based on skill (original logic)
        for agent in agents:
            agent.monthly_wage = agent.skill * self.labor_hours  # Direct skill-based wage
    
    def _update_interest_rates(self):
        """Update interest rates annually."""
        if self.current_step % 12 == 0:
            new_rate = taylor_rule_interest_rate(
                current_rate=self.interest_rate,
                inflation_rate=self.inflation_rate,
                unemployment_rate=self.unemployment_rate
            )
            
            self.interest_rate = new_rate
            self.interest_rate_history.append(new_rate)
            
            # Apply interest to savings
            agents = self.schedule.agents
            for agent in agents:
                interest_earned = agent.wealth * self.interest_rate
                agent.wealth += interest_earned
                agent.interest_earned = interest_earned
    
    def _tax_and_redistribution_phase(self):
        """Tax collection and redistribution."""
        total_tax_collected = 0.0
        agents = self.schedule.agents
        
        for agent in agents:
            if agent.last_income > 0:
                tax_owed = compute_income_tax(agent.last_income, self.tax_brackets, self.tax_rates)
                agent.tax_paid = tax_owed
                agent.wealth -= tax_owed
                total_tax_collected += tax_owed
            else:
                agent.tax_paid = 0.0
        
        # Redistribute equally
        self.government_revenue = total_tax_collected
        self.redistribution_pool = total_tax_collected
        
        if self.n_agents > 0:
            redistribution_per_agent = self.redistribution_pool / self.n_agents
            for agent in agents:
                agent.wealth += redistribution_per_agent
                agent.redistribution_received = redistribution_per_agent
    
    def _collect_data(self):
        """Collect simulation data."""
        agents = self.schedule.agents
        
        # Model-level data
        model_record = {
            'Step': self.current_step,
            'Year': (self.current_step - 1) // 12 + 1,
            'Month': ((self.current_step - 1) % 12) + 1,
            'GDP': sum(agent.last_income for agent in agents),
            'Inflation': self.inflation_rate,
            'Unemployment': self.unemployment_rate,
            'Average_Wage': self.average_wage,
            'Goods_Price': self.goods_price,
            'Interest_Rate': self.interest_rate,
            'Government_Revenue': self.government_revenue,
            'Redistribution': self.redistribution_pool,
            'Goods_Inventory': self.goods_inventory,
            'Total_Wealth': sum(agent.wealth for agent in agents),
            'Gini_Coefficient': self._calculate_gini(),
            'Employment_Rate': 1.0 - self.unemployment_rate,
            'Average_Consumption': sum(getattr(agent, 'actual_consumption', 0) for agent in agents) / self.n_agents
        }
        
        self.model_data.append(model_record)
        
        # Agent-level data (sample every few steps to avoid memory issues)
        if self.current_step % 3 == 0:  # Collect every 3 months
            for agent in agents:
                agent_record = {
                    'Step': self.current_step,
                    'Agent_ID': agent.unique_id,
                    'Wealth': agent.wealth,
                    'Skill': agent.skill,
                    'Job': agent.job,
                    'Last_Work': agent.last_work_decision,
                    'Last_Consumption': agent.last_consumption_decision,
                    'Income': agent.last_income,
                    'Tax_Paid': agent.tax_paid,
                    'Redistribution_Received': agent.redistribution_received
                }
                self.agent_data.append(agent_record)
    
    def _calculate_gini(self):
        """Calculate Gini coefficient."""
        agents = self.schedule.agents
        wealths = [agent.wealth for agent in agents]
        wealths.sort()
        n = len(wealths)
        
        if n == 0 or sum(wealths) == 0:
            return 0.0
        
        cumsum = np.cumsum(wealths)
        return (n + 1 - 2 * sum((n + 1 - i) * w for i, w in enumerate(wealths, 1)) / cumsum[-1]) / n
    
    def _log_progress(self):
        """Log simulation progress."""
        year = (self.current_step - 1) // 12 + 1
        month = ((self.current_step - 1) % 12) + 1
        
        logger.info(f"Step {self.current_step} (Year {year}, Month {month}): "
                   f"GDP=${self._calculate_gdp():.0f}, "
                   f"Unemployment={self.unemployment_rate:.1%}, "
                   f"Inflation={self.inflation_rate:.1%}")
    
    def _calculate_gdp(self):
        """Calculate current GDP."""
        agents = self.schedule.agents
        return sum(agent.last_income for agent in agents)
    
    def get_results_dataframe(self):
        """Get model results as DataFrame."""
        return pd.DataFrame(self.model_data)
    
    def get_agent_dataframe(self):
        """Get agent data as DataFrame."""
        return pd.DataFrame(self.agent_data)
    
    def save_results(self, filepath: str):
        """Save results to Excel file."""
        model_df = self.get_results_dataframe()
        agent_df = self.get_agent_dataframe()
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            model_df.to_excel(writer, sheet_name='Model_Data', index=True)
            agent_df.to_excel(writer, sheet_name='Agent_Data', index=True)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_summary_stats(self):
        """Get summary statistics."""
        model_df = self.get_results_dataframe()
        
        return {
            'simulation_length': self.current_step,
            'final_gdp': model_df['GDP'].iloc[-1] if len(model_df) > 0 else 0,
            'avg_unemployment': model_df['Unemployment'].mean() if len(model_df) > 0 else 0,
            'avg_inflation': model_df['Inflation'].mean() if len(model_df) > 0 else 0,
            'final_gini': model_df['Gini_Coefficient'].iloc[-1] if len(model_df) > 0 else 0,
            'total_agents': self.n_agents,
            'llm_stats': {'decisions_made': 0, 'reflections_made': 0}  # No LLM in simple model
        }