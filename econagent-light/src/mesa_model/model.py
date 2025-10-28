"""
Mesa-based economic simulation model for EconAgent-Light.
Replaces ai-economist foundation with standard Mesa ABM framework.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import mesa
from mesa import Model
from mesa.datacollection import DataCollector

# Simple scheduler for Mesa 3.x (no built-in schedulers)
class SimpleScheduler:
    """Simple random activation scheduler for Mesa 3.x compatibility."""
    
    def __init__(self, model):
        self.model = model
        self.agents = []
    
    def add(self, agent):
        """Add an agent to the scheduler."""
        self.agents.append(agent)
    
    def remove(self, agent):
        """Remove an agent from the scheduler."""
        if agent in self.agents:
            self.agents.remove(agent)
    
    def step(self):
        """Activate all agents in random order."""
        agents_shuffled = self.model.random.sample(self.agents, len(self.agents))
        for agent in agents_shuffled:
            agent.step()
    
    @property
    def agent_count(self):
        return len(self.agents)

# Optional imports for LLM integration
try:
    from ..llm_integration import UnifiedLLMClient
except ImportError:
    UnifiedLLMClient = None

from .utils import (
    compute_income_tax, 
    US_TAX_BRACKETS, 
    US_TAX_RATES_2018,
    pareto_skill_distribution,
    update_wages_and_prices,
    taylor_rule_interest_rate
)

logger = logging.getLogger(__name__)

class EconModel(Model):
    """
    Mesa-based economic simulation model.
    Migrates original ACL24-EconAgent logic to Mesa framework.
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        episode_length: int = 240,  # 20 years * 12 months
        random_seed: Optional[int] = None,
        # Economic parameters from original config
        productivity: float = 1.0,
        skill_change: float = 0.02,
        price_change: float = 0.02,
        max_price_inflation: float = 0.1,
        max_wage_inflation: float = 0.05,
        pareto_param: float = 8.0,
        payment_max_skill_multiplier: float = 950.0,
        labor_hours: int = 168,
        consumption_rate_step: float = 0.02,
        base_interest_rate: float = 0.01,
        # LLM and LightAgent configuration
        llm_client: Optional[Any] = None,
        enable_lightagent: bool = True,
        enable_memory: bool = True,
        enable_tot: bool = True,
        # Simulation parameters
        save_frequency: int = 6,
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
        
        # Economic parameters (from original config.yaml)
        self.productivity = productivity
        self.skill_change = skill_change
        self.price_change = price_change
        self.max_price_inflation = max_price_inflation
        self.max_wage_inflation = max_wage_inflation
        self.pareto_param = pareto_param
        self.payment_max_skill_multiplier = payment_max_skill_multiplier
        self.labor_hours = labor_hours
        self.consumption_rate_step = consumption_rate_step
        self.base_interest_rate = base_interest_rate
        
        # Simulation parameters
        self.save_frequency = save_frequency
        self.log_frequency = log_frequency
        
        # Economic state variables
        self.goods_inventory = 0.0
        self.goods_price = 1.0  # Initial price level
        self.average_wage = 1.0  # Initial wage level
        self.interest_rate = base_interest_rate
        self.inflation_rate = 0.0
        self.unemployment_rate = 0.0
        
        # Economic history for calculations
        self.price_history = [self.goods_price]
        self.wage_history = [self.average_wage]
        self.interest_rate_history = [self.interest_rate]
        
        # Tax system (from original simulate_utils.py)
        self.tax_brackets = US_TAX_BRACKETS.copy()
        self.tax_rates = US_TAX_RATES_2018.copy()
        self.government_revenue = 0.0
        self.redistribution_pool = 0.0
        
        # Initialize LLM and LightAgent (optional)
        self.llm_client = llm_client
        self.enable_lightagent = enable_lightagent and (llm_client is not None)
        self.light_agent_wrapper = None
        
        if not self.enable_lightagent:
            logger.info("Using heuristic agent decisions (no LLM)")
        else:
            logger.info("LightAgent integration enabled")
            try:
                from ..lightagent_integration.light_client import LightAgentWrapper
                self.light_agent_wrapper = LightAgentWrapper(
                    llm_client=self.llm_client,
                    enable_memory=enable_memory,
                    enable_tot=enable_tot,
                    memory_window=1,
                    reflection_frequency=3
                )
            except ImportError:
                logger.warning("LightAgent integration not available, using heuristic decisions")
                self.enable_lightagent = False
        
        # Mesa scheduler
        self.schedule = SimpleScheduler(self)
        
        # Create agents
        self._create_agents()
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Step": "current_step",
                "Year": lambda m: m.current_step // 12 + 1,
                "Month": lambda m: (m.current_step % 12) + 1,
                "GDP": self._calculate_gdp,
                "Inflation": lambda m: m.inflation_rate,
                "Unemployment": lambda m: m.unemployment_rate,
                "Average_Wage": lambda m: m.average_wage,
                "Goods_Price": lambda m: m.goods_price,
                "Interest_Rate": lambda m: m.interest_rate,
                "Government_Revenue": lambda m: m.government_revenue,
                "Redistribution": lambda m: m.redistribution_pool,
                "Goods_Inventory": lambda m: m.goods_inventory,
                "Total_Wealth": self._calculate_total_wealth,
                "Gini_Coefficient": self._calculate_gini,
                "Employment_Rate": self._calculate_employment_rate,
                "Average_Consumption": self._calculate_average_consumption,
                "LLM_Stats": self._get_llm_stats
            },
            agent_reporters={
                "Agent_ID": "unique_id",
                "Wealth": lambda a: a.wealth,
                "Skill": lambda a: a.skill,
                "Job": lambda a: a.job,
                "Last_Work": lambda a: a.last_work_decision,
                "Last_Consumption": lambda a: a.last_consumption_decision,
                "Income": lambda a: a.last_income,
                "Tax_Paid": lambda a: a.tax_paid,
                "Redistribution_Received": lambda a: a.redistribution_received
            }
        )
        
        # Initialize data collection
        self.datacollector.collect(self)
        
        logger.info(f"EconModel initialized: {n_agents} agents, {episode_length} steps")
        if self.light_agent_wrapper:
            logger.info("LightAgent integration enabled")
        else:
            logger.warning("LightAgent integration disabled - using fallback decisions")
    
    def _create_agents(self):
        """Create economic agents with skill distribution."""
        logger.info(f"Creating {self.n_agents} economic agents...")
        
        # Generate skills using Pareto distribution (from original)
        skills = pareto_skill_distribution(
            self.n_agents, 
            self.pareto_param, 
            self.payment_max_skill_multiplier
        )
        
        # Job categories (from original endogenous.py)
        job_categories = ["Engineer", "Teacher", "Doctor", "Lawyer", "Artist", "Manager", "Worker"]
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        
        # Import the agent class
        from .agents import EconAgent
        
        for i in range(self.n_agents):
            agent = EconAgent(
                unique_id=i,
                model=self,
                skill=skills[i],
                initial_wealth=self.random.uniform(100, 1000)  # Random initial wealth
            )
            
            self.schedule.add(agent)
        
        logger.info(f"Created {len(self.schedule.agents)} agents")
    
    def step(self):
        """Execute one simulation step (one month)."""
        self.current_step += 1
        year = (self.current_step - 1) // 12 + 1
        month = ((self.current_step - 1) % 12) + 1
        
        logger.debug(f"Step {self.current_step}: Year {year}, Month {month}")
        
        # 1. Agent decision-making phase
        self._agent_decision_phase()
        
        # 2. Production and consumption simulation
        self._production_phase()
        self._consumption_phase()
        
        # 3. Economic updates
        self._update_wages_and_prices()
        self._update_interest_rates()
        
        # 4. Tax collection and redistribution
        self._tax_and_redistribution_phase()
        
        # 5. Quarterly reflections (every 3 months)
        if self.current_step % 3 == 0:
            self._quarterly_reflection_phase()
        
        # 6. Data collection
        self.datacollector.collect(self)
        
        # 7. Logging and progress
        if self.current_step % self.log_frequency == 0:
            self._log_progress()
        
        # Check if simulation is complete
        if self.current_step >= self.episode_length:
            self.running = False
            logger.info(f"Simulation completed after {self.current_step} steps")
    
    def _agent_decision_phase(self):
        """Phase 1: Agents make work and consumption decisions."""
        logger.debug("Agent decision phase")
        
        # Update economic indicators for agent context
        self._update_economic_indicators()
        
        # Activate all agents for decision-making
        self.schedule.step()
    
    def _production_phase(self):
        """Phase 2: Calculate production based on agent work decisions."""
        total_production = 0.0
        
        for agent in self.schedule.agents:
            if agent.worked_this_month:
                # Production = hours worked * skill * productivity
                production = self.labor_hours * agent.skill * self.productivity
                total_production += production
        
        # Add to goods inventory
        self.goods_inventory += total_production
        
        logger.debug(f"Total production: {total_production:.2f}, Inventory: {self.goods_inventory:.2f}")
    
    def _consumption_phase(self):
        """Phase 3: Process agent consumption decisions."""
        total_consumption_demand = 0.0
        
        # Calculate total consumption demand
        for agent in self.schedule.agents:
            consumption_amount = agent.consumption_spending
            total_consumption_demand += consumption_amount
        
        # Process consumption (limited by inventory)
        actual_consumption = min(total_consumption_demand, self.goods_inventory)
        consumption_ratio = actual_consumption / total_consumption_demand if total_consumption_demand > 0 else 1.0
        
        # Allocate actual consumption to agents
        for agent in self.schedule.agents:
            if agent.consumption_spending > 0:
                actual_agent_consumption = agent.consumption_spending * consumption_ratio
                agent.actual_consumption = actual_agent_consumption
                agent.wealth -= actual_agent_consumption
            else:
                agent.actual_consumption = 0.0
        
        # Update inventory
        self.goods_inventory -= actual_consumption
        self.goods_inventory = max(0.0, self.goods_inventory)  # Can't go negative
        
        logger.debug(f"Consumption demand: {total_consumption_demand:.2f}, Actual: {actual_consumption:.2f}")
    
    def _update_wages_and_prices(self):
        """Phase 4: Update wages and prices using original equations."""
        # Calculate employment and production metrics
        employed_agents = [a for a in self.schedule.agents if a.worked_this_month]
        employment_rate = len(employed_agents) / self.n_agents
        self.unemployment_rate = 1.0 - employment_rate
        
        # Update wages and prices using original logic
        new_wage, new_price = update_wages_and_prices(
            current_wage=self.average_wage,
            current_price=self.goods_price,
            employment_rate=employment_rate,
            inventory_level=self.goods_inventory,
            max_wage_inflation=self.max_wage_inflation,
            max_price_inflation=self.max_price_inflation,
            wage_adjustment_rate=0.05,  # alpha_w from original
            price_adjustment_rate=0.10   # alpha_P from original
        )
        
        # Calculate inflation
        if len(self.price_history) > 0:
            self.inflation_rate = (new_price - self.price_history[-1]) / self.price_history[-1]
        
        # Update values and history
        self.average_wage = new_wage
        self.goods_price = new_price
        self.wage_history.append(new_wage)
        self.price_history.append(new_price)
        
        # Update agent wages based on skill
        for agent in self.schedule.agents:
            agent.monthly_wage = agent.skill * self.average_wage * self.labor_hours
    
    def _update_interest_rates(self):
        """Update interest rates using Taylor rule (annually)."""
        if self.current_step % 12 == 0:  # Annual update
            new_rate = taylor_rule_interest_rate(
                current_rate=self.interest_rate,
                inflation_rate=self.inflation_rate,
                unemployment_rate=self.unemployment_rate,
                natural_rate=0.01,  # rn from original
                inflation_target=0.02,  # pi_t from original
                unemployment_target=0.04,  # un from original
                alpha_pi=0.5,  # from original
                alpha_u=0.5   # from original
            )
            
            self.interest_rate = new_rate
            self.interest_rate_history.append(new_rate)
            
            # Apply interest to agent savings
            for agent in self.schedule.agents:
                interest_earned = agent.wealth * self.interest_rate
                agent.wealth += interest_earned
                agent.interest_earned = interest_earned
    
    def _tax_and_redistribution_phase(self):
        """Phase 5: Collect taxes and redistribute."""
        total_tax_collected = 0.0
        
        # Collect taxes from agents
        for agent in self.schedule.agents:
            if agent.last_income > 0:
                tax_owed = compute_income_tax(agent.last_income, self.tax_brackets, self.tax_rates)
                agent.tax_paid = tax_owed
                agent.wealth -= tax_owed
                total_tax_collected += tax_owed
            else:
                agent.tax_paid = 0.0
        
        # Redistribute equally (from original logic)
        self.government_revenue = total_tax_collected
        self.redistribution_pool = total_tax_collected
        
        if self.n_agents > 0:
            redistribution_per_agent = self.redistribution_pool / self.n_agents
            
            for agent in self.schedule.agents:
                agent.wealth += redistribution_per_agent
                agent.redistribution_received = redistribution_per_agent
        
        logger.debug(f"Tax collected: ${total_tax_collected:.2f}, Redistributed: ${self.redistribution_pool:.2f}")
    
    def _quarterly_reflection_phase(self):
        """Phase 6: Quarterly agent reflections."""
        if not self.light_agent_wrapper:
            return
        
        logger.debug("Quarterly reflection phase")
        
        # Collect last 3 months of data for each agent
        for agent in self.schedule.agents:
            quarterly_data = agent.get_quarterly_data()
            
            if len(quarterly_data) >= 3:  # Have at least 3 months of data
                try:
                    reflection = self.light_agent_wrapper.reflect(
                        agent_profile=agent.get_profile(),
                        quarterly_data=quarterly_data[-3:]  # Last 3 months
                    )
                    agent.last_reflection = reflection
                    
                except Exception as e:
                    logger.warning(f"Reflection failed for agent {agent.unique_id}: {e}")
    
    def _update_economic_indicators(self):
        """Update economic indicators for agent context."""
        # These are used by agents for decision-making context
        pass  # Already updated in other phases
    
    def _log_progress(self):
        """Log simulation progress."""
        year = (self.current_step - 1) // 12 + 1
        month = ((self.current_step - 1) % 12) + 1
        
        logger.info(f"Step {self.current_step} (Year {year}, Month {month}): "
                   f"GDP=${self._calculate_gdp():.0f}, "
                   f"Unemployment={self.unemployment_rate:.1%}, "
                   f"Inflation={self.inflation_rate:.1%}, "
                   f"Price=${self.goods_price:.2f}")
        
        if self.light_agent_wrapper:
            llm_stats = self.light_agent_wrapper.get_stats()
            logger.info(f"LLM Stats: {llm_stats['decisions_made']} decisions, "
                       f"{llm_stats['reflections_made']} reflections")
    
    # Data collection methods
    def _calculate_gdp(self) -> float:
        """Calculate GDP as total production value."""
        total_income = sum(agent.last_income for agent in self.schedule.agents)
        return total_income
    
    def _calculate_total_wealth(self) -> float:
        """Calculate total agent wealth."""
        return sum(agent.wealth for agent in self.schedule.agents)
    
    def _calculate_gini(self) -> float:
        """Calculate Gini coefficient for wealth inequality."""
        wealths = [agent.wealth for agent in self.schedule.agents]
        wealths.sort()
        n = len(wealths)
        
        if n == 0 or sum(wealths) == 0:
            return 0.0
        
        cumsum = np.cumsum(wealths)
        return (n + 1 - 2 * sum((n + 1 - i) * w for i, w in enumerate(wealths, 1)) / cumsum[-1]) / n
    
    def _calculate_employment_rate(self) -> float:
        """Calculate employment rate."""
        employed = sum(1 for agent in self.schedule.agents if agent.worked_this_month)
        return employed / self.n_agents if self.n_agents > 0 else 0.0
    
    def _calculate_average_consumption(self) -> float:
        """Calculate average consumption."""
        total_consumption = sum(getattr(agent, 'actual_consumption', 0) for agent in self.schedule.agents)
        return total_consumption / self.n_agents if self.n_agents > 0 else 0.0
    
    def _get_llm_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        if self.light_agent_wrapper:
            return self.light_agent_wrapper.get_stats()
        return {"decisions_made": 0, "reflections_made": 0}
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get simulation results as pandas DataFrame."""
        return self.datacollector.get_model_vars_dataframe()
    
    def get_agent_dataframe(self) -> pd.DataFrame:
        """Get agent data as pandas DataFrame."""
        return self.datacollector.get_agent_vars_dataframe()
    
    def save_results(self, filepath: str):
        """Save simulation results to file."""
        model_data = self.get_results_dataframe()
        agent_data = self.get_agent_dataframe()
        
        with pd.ExcelWriter(filepath) as writer:
            model_data.to_excel(writer, sheet_name='Model_Data', index=True)
            agent_data.to_excel(writer, sheet_name='Agent_Data', index=True)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the simulation."""
        model_data = self.get_results_dataframe()
        
        return {
            "simulation_length": self.current_step,
            "final_gdp": model_data["GDP"].iloc[-1] if len(model_data) > 0 else 0,
            "avg_unemployment": model_data["Unemployment"].mean() if len(model_data) > 0 else 0,
            "avg_inflation": model_data["Inflation"].mean() if len(model_data) > 0 else 0,
            "final_gini": model_data["Gini_Coefficient"].iloc[-1] if len(model_data) > 0 else 0,
            "total_agents": self.n_agents,
            "llm_stats": self._get_llm_stats()
        }