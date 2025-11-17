"""
Mesa-based agent-based model for stress testing and systemic risk analysis.
"""
from mesa import Model, Agent
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# Import agents - done at end of file to avoid circular imports


class RiskSimulationModel(Model):
    """
    Mesa-based agent-based model for stress testing and systemic risk analysis.
    
    This model simulates interactions between banks and firms under various
    economic scenarios to assess systemic risk propagation.
    """
    
    def __init__(
        self,
        n_banks: int = 10,
        n_firms: int = 50,
        n_workers: int = 0,
        scenario=None,
        random_seed: Optional[int] = None,
        use_llm_agents: bool = False,
        reflection_frequency: int = 3,
        batch_size: int = 64
    ):
        """
        Initialize the risk simulation model.
        
        Args:
            n_banks: Number of bank agents
            n_firms: Number of firm agents
            n_workers: Number of LLM worker agents (0 = disabled)
            scenario: Economic scenario to apply (default: None for baseline)
            random_seed: Random seed for reproducibility
            use_llm_agents: Whether to use LLM-based agents
            reflection_frequency: Months between agent reflections
            batch_size: Batch size for LLM calls
        """
        super().__init__(seed=random_seed)
        self.n_banks = n_banks
        self.n_firms = n_firms
        self.n_workers = n_workers
        self.scenario = scenario
        self.use_llm_agents = use_llm_agents
        self.reflection_frequency = reflection_frequency
        self.batch_size = batch_size
        
        # Set numpy random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Economic state variables
        self.unemployment_rate = 0.04  # 4% baseline
        self.gdp_growth = 0.02  # 2% baseline
        self.interest_rate = 0.03  # 3% baseline
        self.credit_spread = 0.02  # 2% baseline
        self.inflation_rate = 0.03  # 3% baseline
        
        # Track simulation step
        self.current_step = 0
        
        # Create data collector
        self.datacollector = DataCollector(
            model_reporters={
                "system_liquidity": self.compute_system_liquidity,
                "default_rate": self.compute_default_rate,
                "network_stress": self.compute_network_stress,
                "avg_capital_ratio": self.compute_avg_capital_ratio,
                "avg_liquidity_ratio": self.compute_avg_liquidity_ratio,
                "unemployment_rate": lambda m: m.unemployment_rate,
                "gdp_growth": lambda m: m.gdp_growth,
                "interest_rate": lambda m: m.interest_rate,
                "credit_spread": lambda m: m.credit_spread
            },
            agent_reporters={
                "capital_ratio": "capital_ratio",
                "liquidity_ratio": "liquidity_ratio",
                "loan_quality": "loan_quality",
                "agent_type": "agent_type"
            }
        )
        
        # Create agents
        self._create_banks()
        self._create_firms()
        self._create_network()
        
        if n_workers > 0:
            self._create_workers()
        
        logger.info(f"Initialized simulation with {n_banks} banks, {n_firms} firms, and {n_workers} workers (LLM={use_llm_agents})")
    
    def _create_banks(self):
        """Initialize bank agents with balance sheets."""
        for i in range(self.n_banks):
            initial_capital = self.random.uniform(100, 500)
            risk_appetite = self.random.uniform(0.3, 0.7)
            
            bank = BankAgent(
                model=self,
                initial_capital=initial_capital,
                risk_appetite=risk_appetite
            )
    
    def _create_firms(self):
        """Initialize firm agents with borrowing needs."""
        for i in range(self.n_firms):
            borrowing_need = self.random.uniform(10, 100)
            default_probability = self.random.uniform(0.01, 0.05)
            
            firm = FirmAgent(
                model=self,
                borrowing_need=borrowing_need,
                base_default_probability=default_probability
            )
    
    def _create_workers(self):
        """Initialize worker agents with LLM decision-making."""
        if WorkerAgent is None:
            logger.warning("WorkerAgent not available, skipping worker creation")
            return
            
        for i in range(self.n_workers):
            initial_savings = self.random.uniform(3000, 10000)
            wage = self.random.uniform(3000, 6000)
            # Use randint instead of normal for Mesa compatibility
            age = self.random.randint(25, 60)
            
            # Create unique ID manually
            worker_id = 1000 + i  # Start at 1000 to avoid conflicts with banks/firms
            
            worker = WorkerAgent(
                unique_id=worker_id,
                model=self,
                age=age,
                occupation="worker",
                location="City",
                initial_savings=initial_savings,
                wage=wage,
                use_llm=self.use_llm_agents
            )
    
    def _create_network(self):
        """Create lending relationships between banks and firms."""
        banks = [agent for agent in self.agents if isinstance(agent, BankAgent)]
        firms = [agent for agent in self.agents if isinstance(agent, FirmAgent)]
        
        # Each firm gets assigned to 1-3 banks (but not more than available banks)
        for firm in firms:
            max_lenders = min(3, len(banks))
            if max_lenders == 0:
                continue
            n_lenders = self.random.randint(1, max_lenders) if max_lenders > 1 else 1
            firm.lenders = self.random.sample(banks, n_lenders)
            
            # Distribute borrowing need across lenders
            for bank in firm.lenders:
                loan_amount = firm.borrowing_need / n_lenders
                bank.borrowers.append(firm)
                bank.loans += loan_amount
                bank.reserves -= loan_amount
    
    def step(self):
        """Execute one simulation step."""
        # Apply scenario shocks if scenario is defined
        if self.scenario is not None:
            self.scenario.apply_shock(self, self.current_step)
        
        # Agents make decisions - iterate through all agents
        for agent in self.agents:
            agent.step()
        
        # Increment step counter
        self.current_step += 1
        
        # Collect data
        self.datacollector.collect(self)
        
        logger.debug(f"Step {self.current_step}: Default rate = {self.compute_default_rate():.4f}")
    
    def run_simulation(self, n_steps: int = 100) -> pd.DataFrame:
        """
        Run full simulation and return results.
        
        Args:
            n_steps: Number of time steps to simulate
            
        Returns:
            DataFrame with model-level metrics over time
        """
        logger.info(f"Running simulation for {n_steps} steps")
        
        for step in range(n_steps):
            self.step()
        
        results = self.datacollector.get_model_vars_dataframe()
        logger.info(f"Simulation complete. Final default rate: {results['default_rate'].iloc[-1]:.4f}")
        
        return results
    
    def compute_system_liquidity(self) -> float:
        """Compute aggregate system liquidity."""
        banks = [agent for agent in self.agents if isinstance(agent, BankAgent)]
        if not banks:
            return 0.0
        
        total_reserves = sum(bank.reserves for bank in banks)
        total_deposits = sum(bank.deposits for bank in banks)
        
        return total_reserves / total_deposits if total_deposits > 0 else 0.0
    
    def compute_default_rate(self) -> float:
        """Compute proportion of firms in default."""
        firms = [agent for agent in self.agents if isinstance(agent, FirmAgent)]
        if not firms:
            return 0.0
        
        defaulted_firms = sum(1 for firm in firms if firm.is_defaulted)
        return defaulted_firms / len(firms)
    
    def compute_network_stress(self) -> float:
        """Compute network stress indicator."""
        banks = [agent for agent in self.agents if isinstance(agent, BankAgent)]
        if not banks:
            return 0.0
        
        # Stress is measured by proportion of banks below regulatory minimum
        stressed_banks = sum(1 for bank in banks if bank.capital_ratio < 0.08)
        return stressed_banks / len(banks)
    
    def compute_avg_capital_ratio(self) -> float:
        """Compute average capital ratio across banks."""
        banks = [agent for agent in self.agents if isinstance(agent, BankAgent)]
        if not banks:
            return 0.0
        
        return np.mean([bank.capital_ratio for bank in banks])
    
    def compute_avg_liquidity_ratio(self) -> float:
        """Compute average liquidity ratio across banks."""
        banks = [agent for agent in self.agents if isinstance(agent, BankAgent)]
        if not banks:
            return 0.0
        
        return np.mean([bank.liquidity_ratio for bank in banks])


# Import agents after class definition to avoid circular imports
from .agents import BankAgent, FirmAgent

# Import LLM agents if available
try:
    from src.agents.econagent import WorkerAgent, FirmAgentLLM
except ImportError:
    WorkerAgent = None
    FirmAgentLLM = None
    logger.warning("LLM agents not available, LLM mode will be disabled")
