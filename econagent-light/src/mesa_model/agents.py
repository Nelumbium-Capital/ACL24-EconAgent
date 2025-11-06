"""
Economic agents for EconAgent-Light Mesa model.
Simple heuristic agents that replicate basic economic behavior.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

import mesa
from mesa import Agent

logger = logging.getLogger(__name__)


class EconAgent(Agent):
    """
    Economic agent replicating original ACL24-EconAgent behavior.
    """
    def __init__(self, unique_id, model, skill, initial_wealth=100.0):
        super().__init__(model)
        self.unique_id = unique_id
        
        # Original agent attributes from ai-economist
        self.skill = skill
        self.wealth = initial_wealth  # Coin inventory
        self.age = model.random.randint(18, 65)
        self.city = model.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'])
        
        # Job status - start with some unemployment
        if model.random.random() < 0.1:
            self.job = 'Unemployment'
        else:
            jobs = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist', 'Manager', 'Worker']
            self.job = model.random.choice(jobs)
        
        # Economic state variables
        self.monthly_wage = 0.0
        self.last_income = 0.0
        self.worked_this_month = False
        self.consumption_spending = 0.0
        self.actual_consumption = 0.0
        
        # Decision variables (for simple heuristic decisions)
        self.last_work_decision = 0.5
        self.last_consumption_decision = 0.3
        
        # Financial tracking
        self.tax_paid = 0.0
        self.redistribution_received = 0.0
        
        # Calculate initial wage based on skill
        self._update_wage()
    
    def _update_wage(self):
        """Update wage based on skill and market conditions."""
        # Original formula: wage = skill * average_wage * labor_hours
        self.monthly_wage = self.skill * self.model.average_wage * self.model.labor_hours
    
    def step(self):
        """Agent step - make economic decisions."""
        self._update_wage()
        
        # Make economic decisions using heuristic algorithms
        work_decision = self._make_work_decision()
        consumption_decision = self._make_consumption_decision()
        
        # Apply work decision
        self.last_work_decision = work_decision
        self.worked_this_month = self.model.random.random() < work_decision
        
        # Calculate income
        if self.worked_this_month:
            if self.job == 'Unemployment':
                # Get a job
                jobs = ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist', 'Manager', 'Worker']
                self.job = self.model.random.choice(jobs)
            
            self.last_income = self.monthly_wage
            self.wealth += self.last_income
        else:
            self.last_income = 0.0
            # Might become unemployed
            if self.job != 'Unemployment' and self.model.random.random() < 0.05:
                self.job = 'Unemployment'
        
        # Apply consumption decision
        self.last_consumption_decision = consumption_decision
        available_funds = self.wealth
        self.consumption_spending = available_funds * consumption_decision
    
    def _make_work_decision(self):
        """Simple heuristic work decision."""
        if self.job == 'Unemployment':
            # More likely to work if unemployed
            base_propensity = 0.8
        else:
            # Work propensity based on wage/wealth ratio
            if self.wealth > 0:
                base_propensity = min(0.9, self.monthly_wage / (self.wealth * 0.1 + 100))
            else:
                base_propensity = 0.9
        
        # Adjust for market conditions
        unemployment_factor = self.model.unemployment_rate * 0.2  # Work more when unemployment high
        propensity = base_propensity + unemployment_factor
        
        # Add some randomness
        propensity += self.model.random.uniform(-0.1, 0.1)
        
        # Clamp and round to 0.02 steps
        propensity = max(0.0, min(1.0, propensity))
        return round(propensity * 50) / 50.0
    
    def _make_consumption_decision(self):
        """Simple heuristic consumption decision."""
        # Base consumption based on wealth level
        if self.wealth > 1000:
            base_consumption = 0.3  # Rich consume less proportion
        elif self.wealth > 200:
            base_consumption = 0.4  # Middle class
        else:
            base_consumption = 0.6  # Poor consume more proportion
        
        # Adjust for price level
        price_factor = 1.0 / (1.0 + self.model.goods_price - 1.0)
        consumption = base_consumption * price_factor
        
        # Adjust for interest rates (save more when rates high)
        interest_factor = 1.0 - (self.model.interest_rate * 2.0)
        consumption *= max(0.5, interest_factor)
        
        # Add randomness
        consumption += self.model.random.uniform(-0.1, 0.1)
        
        # Clamp and round
        consumption = max(0.0, min(1.0, consumption))
        return round(consumption * 50) / 50.0
    
    def get_financial_summary(self):
        """Get financial summary for analysis."""
        return {
            'wealth': self.wealth,
            'last_income': self.last_income,
            'monthly_wage': self.monthly_wage,
            'tax_paid': self.tax_paid,
            'redistribution_received': self.redistribution_received,
            'consumption_spending': self.consumption_spending,
            'actual_consumption': self.actual_consumption,
            'skill': self.skill,
            'job': self.job
        }