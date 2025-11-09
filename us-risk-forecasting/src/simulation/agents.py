"""
Agent classes for the risk simulation model.
"""
from mesa import Agent
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class BankAgent(Agent):
    """
    Bank agent with balance sheet, lending decisions, and risk management.
    
    Banks maintain a balance sheet with capital, deposits, loans, and reserves.
    They make lending decisions based on capital ratios and risk appetite.
    """
    
    def __init__(
        self,
        model,
        initial_capital: float,
        risk_appetite: float
    ):
        """
        Initialize a bank agent.
        
        Args:
            model: Reference to the model
            initial_capital: Initial capital amount
            risk_appetite: Risk appetite parameter (0-1, higher = more aggressive)
        """
        super().__init__(model)
        
        # Balance sheet components
        self.capital = initial_capital
        self.deposits = initial_capital * 5  # 5x leverage
        self.loans = initial_capital * 4  # Initial loan portfolio
        self.reserves = initial_capital  # Liquid reserves
        
        # Risk parameters
        self.risk_appetite = risk_appetite
        
        # Relationships
        self.borrowers: List['FirmAgent'] = []
        
        # Agent type for reporting
        self.agent_type = "bank"
        
        # Track loan quality
        self.loan_quality = 1.0  # 1.0 = perfect, 0.0 = all defaulted
    
    @property
    def capital_ratio(self) -> float:
        """Compute capital adequacy ratio (capital / risk-weighted assets)."""
        if self.loans <= 0:
            return 1.0
        return self.capital / self.loans
    
    @property
    def liquidity_ratio(self) -> float:
        """Compute liquidity coverage ratio (reserves / deposits)."""
        if self.deposits <= 0:
            return 1.0
        return self.reserves / self.deposits
    
    def step(self):
        """Execute bank's decision-making for this period."""
        # 1. Assess loan portfolio quality
        self._assess_loan_quality()
        
        # 2. Make lending decisions
        self._make_lending_decisions()
        
        # 3. Manage liquidity
        self._manage_liquidity()
        
        # 4. Update capital
        self._update_capital()
    
    def _assess_loan_quality(self):
        """Evaluate loan portfolio and recognize losses."""
        if not self.borrowers or self.loans <= 0:
            self.loan_quality = 1.0
            return
        
        # Count actual defaults from borrower firms
        defaulted_count = sum(1 for firm in self.borrowers if firm.is_defaulted)
        total_borrowers = len(self.borrowers)
        
        if total_borrowers > 0:
            actual_default_rate = defaulted_count / total_borrowers
        else:
            actual_default_rate = 0.0
        
        # Calculate loan losses based on actual defaults (not entire portfolio!)
        # Assume 60% loss given default (LGD)
        lgd = 0.60
        loan_losses = self.loans * actual_default_rate * lgd
        
        # Update balance sheet
        self.capital -= loan_losses
        self.loans -= loan_losses
        
        # Update loan quality metric
        self.loan_quality = max(0.0, 1.0 - actual_default_rate)
        
        # Ensure capital doesn't go negative (bank failure)
        if self.capital < 0:
            logger.debug(f"Bank {self.unique_id} has negative capital: {self.capital:.2f}")
            self.capital = 0.01  # Minimal capital to avoid division by zero
    
    def _make_lending_decisions(self):
        """Decide on new loan origination."""
        # Only lend if capital ratio is above regulatory minimum
        if self.capital_ratio < 0.08:
            logger.debug(f"Bank {self.unique_id} below regulatory minimum, not lending")
            return
        
        # Adjust lending based on capital ratio and risk appetite
        if self.capital_ratio > 0.10:  # Above regulatory minimum with buffer
            # Calculate new lending capacity
            excess_capital = self.capital - (self.loans * 0.08)
            new_loans = excess_capital * self.risk_appetite * 0.5
            
            # Ensure we have reserves to fund the loans
            available_reserves = max(0, self.reserves - self.deposits * 0.1)
            new_loans = min(new_loans, available_reserves)
            
            if new_loans > 0:
                self.loans += new_loans
                self.reserves -= new_loans
                logger.debug(f"Bank originated {new_loans:.2f} in new loans")
    
    def _manage_liquidity(self):
        """Manage liquidity position."""
        # Target liquidity ratio
        target_liquidity = 0.15
        
        current_liquidity = self.liquidity_ratio
        
        if current_liquidity < target_liquidity:
            # Need to raise liquidity - reduce loans or raise deposits
            liquidity_gap = (target_liquidity * self.deposits) - self.reserves
            
            # Try to raise deposits (simplified - just increase deposits)
            deposit_increase = liquidity_gap * 0.5
            self.deposits += deposit_increase
            self.reserves += deposit_increase
            
            logger.debug(f"Bank raised {deposit_increase:.2f} in deposits")
    
    def _update_capital(self):
        """Update capital based on earnings."""
        # Simple earnings model: interest income minus interest expense
        interest_income = self.loans * self.model.interest_rate
        interest_expense = self.deposits * (self.model.interest_rate * 0.5)
        
        net_income = interest_income - interest_expense
        
        # Add to capital (retained earnings)
        self.capital += net_income * 0.5  # 50% retention ratio


class FirmAgent(Agent):
    """
    Firm agent with borrowing needs and default probability.
    
    Firms borrow from banks and may default based on economic conditions.
    """
    
    def __init__(
        self,
        model,
        borrowing_need: float,
        base_default_probability: float
    ):
        """
        Initialize a firm agent.
        
        Args:
            model: Reference to the model
            borrowing_need: Amount of borrowing required
            base_default_probability: Baseline probability of default
        """
        super().__init__(model)
        
        self.borrowing_need = borrowing_need
        self.base_default_probability = base_default_probability
        
        # State
        self.is_defaulted = False
        self.debt_outstanding = borrowing_need
        
        # Relationships
        self.lenders: List[BankAgent] = []
        
        # Agent type for reporting
        self.agent_type = "firm"
        
        # Placeholder for loan quality (not used but needed for reporting)
        self.loan_quality = None
        self.capital_ratio = None
        self.liquidity_ratio = None
    
    def step(self):
        """Execute firm's decision-making for this period."""
        if self.is_defaulted:
            return  # Already defaulted, no further actions
        
        # Determine if firm defaults this period
        self._evaluate_default()
        
        # Make debt payments if not defaulted
        if not self.is_defaulted:
            self._make_debt_payments()
    
    def _evaluate_default(self):
        """Determine if firm defaults based on economic conditions."""
        # Adjust default probability based on economic conditions
        unemployment = self.model.unemployment_rate
        gdp_growth = self.model.gdp_growth
        credit_spread = self.model.credit_spread
        
        # More realistic stress factors (smaller multipliers)
        # Base default rate is annual, so divide by 12 for monthly
        monthly_base_default = self.base_default_probability / 12
        
        # Stress multiplier (more moderate)
        unemployment_stress = max(0, (unemployment - 0.04) * 2)  # Reduced from 5
        credit_stress = max(0, (credit_spread - 0.02) * 3)  # Reduced from 10
        gdp_benefit = max(0, (gdp_growth - 0.02) * 1)  # Reduced from 5
        
        stress_factor = 1.0 + unemployment_stress + credit_stress - gdp_benefit
        
        # Monthly default probability (much more realistic)
        adjusted_default_prob = min(
            monthly_base_default * max(stress_factor, 0.5),
            0.05  # Cap at 5% per month (60% annualized in extreme stress)
        )
        
        # Random draw to determine default
        if self.model.random.random() < adjusted_default_prob:
            self.is_defaulted = True
            logger.debug(f"Firm {self.unique_id} defaulted (prob={adjusted_default_prob:.4f})")
            
            # Notify lenders of default
            for lender in self.lenders:
                # Lender will recognize loss in their assessment step
                pass
    
    def _make_debt_payments(self):
        """Make periodic debt payments to lenders."""
        # Simple model: pay interest on outstanding debt
        interest_payment = self.debt_outstanding * self.model.interest_rate
        
        # Reduce debt slightly (amortization)
        principal_payment = self.debt_outstanding * 0.02  # 2% per period
        
        self.debt_outstanding -= principal_payment
        
        # In a more complex model, these payments would flow to lender banks
        logger.debug(f"Firm made payment: {interest_payment + principal_payment:.2f}")
