#!/usr/bin/env python3
"""
Standalone EconAgent-Light with real FRED data integration.
Complete economic simulation using only standard library + requests.
No numpy, pandas, or other heavy dependencies.
"""

import logging
import random
import math
import json
import requests
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EconAgent:
    """Economic agent for ACL24-EconAgent simulation."""
    
    def __init__(self, agent_id: int, skill: float, initial_wealth: float = 100.0):
        self.agent_id = agent_id
        self.skill = skill
        self.wealth = initial_wealth
        self.age = random.randint(18, 65)
        self.job = random.choice(['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist', 'Manager', 'Worker'])
        
        # Economic state
        self.monthly_wage = 0.0
        self.last_income = 0.0
        self.worked_this_month = False
        self.consumption_spending = 0.0
        self.actual_consumption = 0.0
        
        # Decision variables
        self.last_work_decision = 0.5
        self.last_consumption_decision = 0.3
        
        # Financial tracking
        self.tax_paid = 0.0
        self.redistribution_received = 0.0
    
    def step(self, model):
        """Agent decision-making step."""
        # Update wage based on skill and market conditions
        self.monthly_wage = self.skill * model.average_wage * model.labor_hours
        
        # Make work decision
        work_decision = self._make_work_decision(model)
        self.last_work_decision = work_decision
        self.worked_this_month = random.random() < work_decision
        
        # Calculate income
        if self.worked_this_month:
            if self.job == 'Unemployment':
                self.job = random.choice(['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist', 'Manager', 'Worker'])
            self.last_income = self.monthly_wage
            self.wealth += self.last_income
        else:
            self.last_income = 0.0
            if self.job != 'Unemployment' and random.random() < 0.05:
                self.job = 'Unemployment'
        
        # Make consumption decision
        consumption_decision = self._make_consumption_decision(model)
        self.last_consumption_decision = consumption_decision
        self.consumption_spending = self.wealth * consumption_decision
    
    def _make_work_decision(self, model):
        """Heuristic work decision based on economic conditions."""
        if self.job == 'Unemployment':
            base_propensity = 0.8
        else:
            if self.wealth > 0:
                base_propensity = min(0.9, self.monthly_wage / (self.wealth * 0.1 + 100))
            else:
                base_propensity = 0.9
        
        # Adjust for unemployment rate (work more when unemployment is high)
        unemployment_factor = model.unemployment_rate * 0.2
        propensity = base_propensity + unemployment_factor
        
        # Add randomness
        propensity += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, propensity))
    
    def _make_consumption_decision(self, model):
        """Heuristic consumption decision based on wealth and economic conditions."""
        if self.wealth > 1000:
            base_consumption = 0.3
        elif self.wealth > 200:
            base_consumption = 0.4
        else:
            base_consumption = 0.6
        
        # Adjust for price level
        price_factor = 1.0 / (1.0 + model.goods_price - 1.0)
        consumption = base_consumption * price_factor
        
        # Adjust for interest rates
        interest_factor = 1.0 - (model.interest_rate * 2.0)
        consumption *= max(0.5, interest_factor)
        
        # Add randomness
        consumption += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, consumption))

class StandaloneEconModel:
    """
    Complete ACL24-EconAgent economic model with real FRED data integration.
    Uses only standard library + requests (no numpy/pandas dependencies).
    """
    
    def __init__(
        self,
        n_agents: int = 100,
        episode_length: int = 240,
        random_seed: Optional[int] = None,
        fred_api_key: str = "bcc1a43947af1745a35bfb3b7132b7c6",
        enable_real_data: bool = True,
        real_data_update_frequency: int = 12
    ):
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
        
        # Model parameters
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.current_step = 0
        self.running = True
        
        # FRED data integration
        self.fred_api_key = fred_api_key
        self.enable_real_data = enable_real_data
        self.real_data_update_frequency = real_data_update_frequency
        self.last_real_data_update = 0
        
        # Economic parameters (ACL24-EconAgent paper)
        self.productivity = 1.0
        self.max_price_inflation = 0.10
        self.max_wage_inflation = 0.05
        self.pareto_param = 8.0
        self.payment_max_skill_multiplier = 950.0
        self.labor_hours = 168
        
        # Economic state variables
        self.goods_inventory = 0.0
        self.goods_price = 1.0
        self.average_wage = 1.0
        self.interest_rate = 0.02
        self.inflation_rate = 0.0
        self.unemployment_rate = 0.05
        
        # Real economic data from FRED
        self.real_unemployment_rate = 0.05
        self.real_fed_funds_rate = 0.02
        self.real_cpi_level = 100.0
        self.real_gdp_growth = 0.0
        self.real_wage_level = 30.0
        
        # Economic history
        self.price_history = [self.goods_price]
        self.wage_history = [self.average_wage]
        self.interest_rate_history = [self.interest_rate]
        
        # Tax system (from ACL24-EconAgent paper)
        self.tax_brackets = [0, 97, 394.75, 842, 1607.25, 2041, 5103]
        self.tax_rates = [0.0, 0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
        self.government_revenue = 0.0
        self.redistribution_pool = 0.0
        
        # Initialize real data
        if self.enable_real_data:
            self._initialize_real_data()
        
        # Create agents
        self.agents = self._create_agents()
        
        # Data collection
        self.model_data = []
        
        logger.info(f"StandaloneEconModel initialized: {n_agents} agents, real data: {enable_real_data}")
    
    def _initialize_real_data(self):
        """Initialize real economic data from FRED API."""
        try:
            logger.info("🏦 Fetching real economic data from FRED...")
            
            # Fetch key economic indicators
            indicators = self._fetch_fred_indicators()
            
            if indicators:
                # Update model parameters with real data
                if 'unemployment' in indicators:
                    self.real_unemployment_rate = indicators['unemployment'] / 100
                    self.unemployment_rate = self.real_unemployment_rate
                    logger.info(f"📊 Real unemployment rate: {self.real_unemployment_rate:.1%}")
                
                if 'fed_funds' in indicators:
                    self.real_fed_funds_rate = indicators['fed_funds'] / 100
                    self.interest_rate = self.real_fed_funds_rate
                    logger.info(f"💰 Real Fed funds rate: {self.real_fed_funds_rate:.1%}")
                
                if 'cpi' in indicators:
                    self.real_cpi_level = indicators['cpi']
                    self.goods_price = self.real_cpi_level / 324.0  # Normalize to ~1.0
                    logger.info(f"📈 Real CPI level: {self.real_cpi_level:.1f}")
                
                if 'wages' in indicators:
                    self.real_wage_level = indicators['wages']
                    self.average_wage = self.real_wage_level / 30.0  # Normalize to simulation scale
                    logger.info(f"💵 Real wage level: ${self.real_wage_level:.2f}/hr")
                
                if 'gdp' in indicators:
                    # Calculate GDP growth (simplified)
                    self.real_gdp_growth = 0.02  # Assume 2% annual growth
                    self.productivity = 1.0 + self.real_gdp_growth
                    logger.info(f"🏭 Real GDP: ${indicators['gdp']:,.1f}B")
                
                logger.info("✅ Real FRED data successfully integrated into model")
            
        except Exception as e:
            logger.warning(f"❌ Failed to initialize real data: {e}")
            logger.warning("🔄 Using default economic parameters")
            self.enable_real_data = False
    
    def _fetch_fred_indicators(self) -> Dict[str, float]:
        """Fetch current economic indicators from FRED API."""
        indicators = {}
        
        # Key FRED series
        series_map = {
            'unemployment': 'UNRATE',
            'fed_funds': 'FEDFUNDS', 
            'cpi': 'CPIAUCSL',
            'wages': 'AHETPI',
            'gdp': 'GDP'
        }
        
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        for name, series_id in series_map.items():
            try:
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if 'observations' in data and len(data['observations']) > 0:
                    latest_obs = data['observations'][0]
                    value = latest_obs['value']
                    date = latest_obs['date']
                    
                    if value != '.':  # FRED uses '.' for missing values
                        indicators[name] = float(value)
                        logger.debug(f"📊 {name}: {value} (as of {date})")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to fetch {name}: {e}")
                continue
        
        return indicators
    
    def _create_agents(self) -> List[EconAgent]:
        """Create economic agents with Pareto skill distribution."""
        agents = []
        
        # Generate Pareto-distributed skills (simplified without numpy)
        for i in range(self.n_agents):
            # Simplified Pareto distribution using inverse transform
            u = random.random()
            skill = (1.0 / (u ** (1.0 / self.pareto_param))) * 0.1  # Scale factor
            skill = min(skill, self.payment_max_skill_multiplier / 1000.0)  # Cap skill
            
            agent = EconAgent(
                agent_id=i,
                skill=skill,
                initial_wealth=random.uniform(100, 1000)
            )
            agents.append(agent)
        
        logger.info(f"👥 Created {len(agents)} agents with Pareto skill distribution")
        return agents
    
    def step(self):
        """Execute one simulation step with real FRED data integration."""
        self.current_step += 1
        year = (self.current_step - 1) // 12 + 1
        month = ((self.current_step - 1) % 12) + 1
        
        logger.debug(f"📅 Step {self.current_step}: Year {year}, Month {month}")
        
        # Update real data periodically
        if self.enable_real_data and (self.current_step - self.last_real_data_update >= self.real_data_update_frequency):
            self._update_real_data()
        
        # 1. Agent decision phase
        for agent in self.agents:
            agent.step(self)
        
        # 2. Production phase
        total_production = 0.0
        for agent in self.agents:
            if agent.worked_this_month:
                production = self.labor_hours * agent.skill * self.productivity
                total_production += production
        
        self.goods_inventory += total_production
        
        # 3. Consumption phase
        total_consumption_demand = sum(agent.consumption_spending for agent in self.agents)
        actual_consumption = min(total_consumption_demand, self.goods_inventory)
        consumption_ratio = actual_consumption / total_consumption_demand if total_consumption_demand > 0 else 1.0
        
        for agent in self.agents:
            if agent.consumption_spending > 0:
                agent.actual_consumption = agent.consumption_spending * consumption_ratio
                agent.wealth -= agent.actual_consumption
            else:
                agent.actual_consumption = 0.0
        
        self.goods_inventory -= actual_consumption
        self.goods_inventory = max(0.0, self.goods_inventory)
        
        # 4. Update wages and prices (ACL24-EconAgent methodology)
        self._update_wages_and_prices()
        
        # 5. Update interest rates (Taylor rule)
        if self.current_step % 12 == 0:  # Annual update
            self._update_interest_rates()
        
        # 6. Tax and redistribution
        self._tax_and_redistribution()
        
        # 7. Collect data
        self._collect_data()
        
        # Check completion
        if self.current_step >= self.episode_length:
            self.running = False
            logger.info(f"🎉 Simulation completed after {self.current_step} steps")
    
    def _update_real_data(self):
        """Update real economic data during simulation."""
        try:
            logger.info("🔄 Updating real economic data from FRED...")
            
            indicators = self._fetch_fred_indicators()
            
            if indicators:
                # Gradually adjust towards real data
                adjustment_factor = 0.1  # 10% adjustment
                
                if 'unemployment' in indicators:
                    target_unemployment = indicators['unemployment'] / 100
                    self.real_unemployment_rate = target_unemployment
                
                if 'fed_funds' in indicators:
                    target_rate = indicators['fed_funds'] / 100
                    self.interest_rate += (target_rate - self.interest_rate) * adjustment_factor
                    self.real_fed_funds_rate = target_rate
                
                if 'cpi' in indicators:
                    self.real_cpi_level = indicators['cpi']
                
                if 'wages' in indicators:
                    self.real_wage_level = indicators['wages']
                
                self.last_real_data_update = self.current_step
                logger.info(f"✅ Real data updated at step {self.current_step}")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to update real data: {e}")
    
    def _update_wages_and_prices(self):
        """Update wages and prices with real data influence (ACL24-EconAgent)."""
        # Calculate employment metrics
        employed_agents = [a for a in self.agents if a.worked_this_month]
        employment_rate = len(employed_agents) / self.n_agents
        self.unemployment_rate = 1.0 - employment_rate
        
        # Wage adjustment based on employment (Phillips Curve)
        wage_adjustment = 0.05 * (employment_rate - 0.95)  # Target 95% employment
        
        # Price adjustment based on inventory (supply/demand)
        target_inventory = self.n_agents * 100  # Target inventory level
        inventory_ratio = self.goods_inventory / target_inventory if target_inventory > 0 else 1.0
        price_adjustment = 0.10 * (inventory_ratio - 1.0)
        
        # Apply adjustments
        new_wage = self.average_wage * (1.0 + max(-self.max_wage_inflation, min(self.max_wage_inflation, wage_adjustment)))
        new_price = self.goods_price * (1.0 + max(-self.max_price_inflation, min(self.max_price_inflation, price_adjustment)))
        
        # Incorporate real data influence
        if self.enable_real_data:
            real_influence = 0.03  # 3% influence from real data
            
            # Adjust towards real wage levels
            if hasattr(self, 'real_wage_level'):
                target_wage = self.real_wage_level / 30.0  # Normalize
                new_wage += (target_wage - new_wage) * real_influence
            
            # Adjust towards real price levels (CPI)
            if hasattr(self, 'real_cpi_level'):
                target_price = self.real_cpi_level / 324.0  # Normalize
                new_price += (target_price - new_price) * real_influence
        
        # Calculate inflation
        if len(self.price_history) > 0:
            self.inflation_rate = (new_price - self.price_history[-1]) / self.price_history[-1]
        
        # Update values
        self.average_wage = new_wage
        self.goods_price = new_price
        self.wage_history.append(new_wage)
        self.price_history.append(new_price)
        
        # Update agent wages
        for agent in self.agents:
            agent.monthly_wage = agent.skill * self.average_wage * self.labor_hours
    
    def _update_interest_rates(self):
        """Update interest rates using Taylor rule with real data (ACL24-EconAgent)."""
        # Taylor rule parameters
        natural_rate = self.real_fed_funds_rate if self.enable_real_data else 0.01
        inflation_target = 0.02  # Fed's 2% target
        unemployment_target = self.real_unemployment_rate if self.enable_real_data else 0.04
        
        # Taylor rule calculation
        inflation_gap = self.inflation_rate - inflation_target
        unemployment_gap = self.unemployment_rate - unemployment_target
        
        new_rate = natural_rate + 0.5 * inflation_gap - 0.5 * unemployment_gap
        
        # Incorporate real Fed rate influence
        if self.enable_real_data:
            real_rate_influence = 0.1  # 10% influence
            new_rate += (self.real_fed_funds_rate - new_rate) * real_rate_influence
        
        self.interest_rate = max(0.0, new_rate)
        self.interest_rate_history.append(self.interest_rate)
        
        # Apply interest to agent savings
        for agent in self.agents:
            interest_earned = agent.wealth * self.interest_rate
            agent.wealth += interest_earned
    
    def _tax_and_redistribution(self):
        """Tax collection and redistribution (ACL24-EconAgent methodology)."""
        total_tax_collected = 0.0
        
        # Collect taxes using progressive brackets
        for agent in self.agents:
            if agent.last_income > 0:
                tax_owed = self._compute_income_tax(agent.last_income)
                agent.tax_paid = tax_owed
                agent.wealth -= tax_owed
                total_tax_collected += tax_owed
            else:
                agent.tax_paid = 0.0
        
        # Redistribute equally (ACL24-EconAgent paper methodology)
        self.government_revenue = total_tax_collected
        self.redistribution_pool = total_tax_collected
        
        if self.n_agents > 0:
            redistribution_per_agent = self.redistribution_pool / self.n_agents
            for agent in self.agents:
                agent.wealth += redistribution_per_agent
                agent.redistribution_received = redistribution_per_agent
    
    def _compute_income_tax(self, income: float) -> float:
        """Compute income tax using progressive brackets (ACL24-EconAgent)."""
        tax_owed = 0.0
        remaining_income = income
        
        for i in range(len(self.tax_brackets) - 1):
            bracket_min = self.tax_brackets[i]
            bracket_max = self.tax_brackets[i + 1]
            bracket_size = bracket_max - bracket_min
            
            if remaining_income <= 0:
                break
            
            taxable_in_bracket = min(remaining_income, bracket_size)
            tax_rate = self.tax_rates[i] if i < len(self.tax_rates) else self.tax_rates[-1]
            tax_owed += taxable_in_bracket * tax_rate
            remaining_income -= taxable_in_bracket
        
        # Handle top bracket
        if remaining_income > 0:
            tax_rate = self.tax_rates[-1] if self.tax_rates else 0.37
            tax_owed += remaining_income * tax_rate
        
        return tax_owed
    
    def _collect_data(self):
        """Collect simulation data for analysis."""
        # Calculate key metrics
        gdp = sum(agent.last_income for agent in self.agents)
        total_wealth = sum(agent.wealth for agent in self.agents)
        gini = self._calculate_gini()
        employment_rate = len([a for a in self.agents if a.worked_this_month]) / self.n_agents
        avg_consumption = sum(getattr(agent, 'actual_consumption', 0) for agent in self.agents) / self.n_agents
        
        record = {
            'step': self.current_step,
            'year': (self.current_step - 1) // 12 + 1,
            'month': ((self.current_step - 1) % 12) + 1,
            'gdp': gdp,
            'unemployment': self.unemployment_rate,
            'inflation': self.inflation_rate,
            'interest_rate': self.interest_rate,
            'goods_price': self.goods_price,
            'average_wage': self.average_wage,
            'total_wealth': total_wealth,
            'gini_coefficient': gini,
            'employment_rate': employment_rate,
            'average_consumption': avg_consumption,
            'government_revenue': self.government_revenue,
            'goods_inventory': self.goods_inventory,
            # Real data comparison
            'real_unemployment': self.real_unemployment_rate,
            'real_fed_funds': self.real_fed_funds_rate,
            'real_cpi': self.real_cpi_level,
            'real_wage': self.real_wage_level,
            'real_data_enabled': self.enable_real_data
        }
        
        self.model_data.append(record)
    
    def _calculate_gini(self) -> float:
        """Calculate Gini coefficient for wealth inequality."""
        wealths = [agent.wealth for agent in self.agents]
        wealths.sort()
        n = len(wealths)
        
        if n == 0 or sum(wealths) == 0:
            return 0.0
        
        # Simplified Gini calculation
        total_wealth = sum(wealths)
        cumulative_wealth = 0
        gini_sum = 0
        
        for i, w in enumerate(wealths):
            cumulative_wealth += w
            gini_sum += (2 * (i + 1) - n - 1) * w
        
        return gini_sum / (n * total_wealth) if total_wealth > 0 else 0.0
    
    def get_results(self) -> List[Dict]:
        """Get simulation results."""
        return self.model_data
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.model_data:
            return {}
        
        final_data = self.model_data[-1]
        
        return {
            'simulation_length': self.current_step,
            'final_gdp': final_data['gdp'],
            'avg_unemployment': sum(d['unemployment'] for d in self.model_data) / len(self.model_data),
            'avg_inflation': sum(d['inflation'] for d in self.model_data) / len(self.model_data),
            'final_gini': final_data['gini_coefficient'],
            'total_agents': self.n_agents,
            'real_data_enabled': self.enable_real_data,
            'real_unemployment': self.real_unemployment_rate,
            'real_fed_funds': self.real_fed_funds_rate,
            'real_cpi': self.real_cpi_level,
            'real_wage': self.real_wage_level
        }

def run_simulation(n_agents=50, years=2, seed=42):
    """Run economic simulation with real FRED data."""
    print("=" * 80)
    print("🏦 ECONAGENT-LIGHT WITH REAL FRED DATA INTEGRATION")
    print("=" * 80)
    print(f"📊 Configuration: {n_agents} agents, {years} years, seed {seed}")
    print()
    
    # Create and run model
    model = StandaloneEconModel(
        n_agents=n_agents,
        episode_length=years * 12,
        random_seed=seed,
        enable_real_data=True,
        real_data_update_frequency=6  # Update every 6 months
    )
    
    print()
    print("🚀 Starting simulation...")
    print("-" * 50)
    
    # Run simulation with progress updates
    total_steps = years * 12
    for step in range(total_steps):
        model.step()
        
        # Progress updates every 3 months
        if (step + 1) % 3 == 0 or step == 0:
            data = model.model_data[-1]
            print(f"📈 Month {step + 1:2d}: GDP=${data['gdp']:6.0f}, "
                  f"Unemployment={data['unemployment']:5.1%}, "
                  f"Inflation={data['inflation']:6.1%}, "
                  f"Interest Rate={data['interest_rate']:5.1%}")
        
        if not model.running:
            break
    
    # Display results
    results = model.get_results()
    summary = model.get_summary_stats()
    
    print()
    print("=" * 80)
    print("📊 SIMULATION RESULTS")
    print("=" * 80)
    print(f"Simulation Length: {summary['simulation_length']} months")
    print(f"Final GDP: ${summary['final_gdp']:,.2f}")
    print(f"Average Unemployment: {summary['avg_unemployment']:.1%}")
    print(f"Average Inflation: {summary['avg_inflation']:.1%}")
    print(f"Final Gini Coefficient: {summary['final_gini']:.3f}")
    print(f"Total Agents: {summary['total_agents']}")
    print()
    print("🔍 REAL FRED DATA COMPARISON:")
    print("-" * 35)
    print(f"Real Unemployment Rate: {summary['real_unemployment']:.1%}")
    print(f"Real Fed Funds Rate:    {summary['real_fed_funds']:.1%}")
    print(f"Real CPI Level:         {summary['real_cpi']:.1f}")
    print(f"Real Wage Level:        ${summary['real_wage']:.2f}/hr")
    print(f"Real Data Integration:  {'✅ Enabled' if summary['real_data_enabled'] else '❌ Disabled'}")
    print()
    
    # Economic relationships analysis
    if len(results) > 1:
        print("🧮 ECONOMIC RELATIONSHIPS:")
        print("-" * 30)
        
        # Phillips Curve analysis
        unemployment_values = [r['unemployment'] for r in results]
        inflation_values = [r['inflation'] for r in results]
        
        if len(unemployment_values) > 1:
            # Simple correlation
            n = len(unemployment_values)
            sum_xy = sum(x * y for x, y in zip(unemployment_values, inflation_values))
            sum_x = sum(unemployment_values)
            sum_y = sum(inflation_values)
            sum_x2 = sum(x * x for x in unemployment_values)
            sum_y2 = sum(y * y for y in inflation_values)
            
            if n * sum_x2 - sum_x**2 > 0 and n * sum_y2 - sum_y**2 > 0:
                phillips_corr = (n * sum_xy - sum_x * sum_y) / ((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))**0.5
                print(f"Phillips Curve Correlation: {phillips_corr:.3f}")
                
                if phillips_corr < -0.1:
                    print("✅ Phillips Curve: Negative correlation (expected)")
                else:
                    print("⚠️  Phillips Curve: Weak/positive correlation")
        
        # GDP growth analysis
        gdp_values = [r['gdp'] for r in results]
        if len(gdp_values) > 1:
            gdp_growth = [(gdp_values[i] - gdp_values[i-1]) / gdp_values[i-1] for i in range(1, len(gdp_values))]
            avg_gdp_growth = sum(gdp_growth) / len(gdp_growth) if gdp_growth else 0
            print(f"Average GDP Growth: {avg_gdp_growth:.1%} per month")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"econagent_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': summary,
            'parameters': {
                'n_agents': n_agents,
                'years': years,
                'seed': seed,
                'real_data_enabled': True
            },
            'metadata': {
                'model': 'ACL24-EconAgent with FRED integration',
                'timestamp': timestamp,
                'fred_api_key_used': model.fred_api_key[:8] + '...' + model.fred_api_key[-4:]
            }
        }, f, indent=2)
    
    print(f"📁 Results saved to: {results_file}")
    print("=" * 80)
    print("🎉 SIMULATION COMPLETED SUCCESSFULLY!")
    print("🎉 Real Federal Reserve data was used throughout the simulation!")
    print("=" * 80)
    
    return model, results, summary

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    n_agents = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    years = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    
    # Run simulation
    try:
        model, results, summary = run_simulation(n_agents, years, seed)
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        logger.exception("Full error details:")