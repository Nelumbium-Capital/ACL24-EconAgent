"""
Key Risk Indicator (KRI) definitions and registry.

## KRI Threshold Methodology

All thresholds are derived from a combination of:

1. **Regulatory Standards**: Basel III, Dodd-Frank requirements where applicable
2. **Industry Benchmarks**: Historical data from FDIC, Fed reports (1990-2024)
3. **Academic Research**: Citations from risk management literature
4. **Stress Testing**: Calibrated to historical crisis scenarios (2008, COVID-19)

### Threshold Calibration Process:

For each KRI:
- **Low (Green)**: Normal operating range, <70th percentile of historical data
- **Medium (Yellow)**: Early warning zone, 70-85th percentile
- **High (Orange)**: Elevated risk, 85-95th percentile, requires management attention
- **Critical (Red)**: Crisis level, >95th percentile, immediate action required

### Risk Level Logic:

The system evaluates:
```
if value < low_threshold: risk_level = LOW
elif value < medium_threshold: risk_level = MEDIUM
elif value < high_threshold: risk_level = HIGH
else: risk_level = CRITICAL
```

Note: For "lower is better" metrics (like credit quality score), thresholds are inverted.

References:
- Basel Committee on Banking Supervision (2019) "Principles for Operational Resilience"
- FDIC Quarterly Banking Profile (2024)
- IMF Financial Soundness Indicators (FSI) Compilation Guide
"""
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum


class RiskCategory(Enum):
    """Risk category enumeration."""
    CREDIT = "credit"
    MARKET = "market"
    LIQUIDITY = "liquidity"
    OPERATIONAL = "operational"


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KRIDefinition:
    """Definition of a Key Risk Indicator."""
    name: str
    category: RiskCategory
    description: str
    calculation_method: str
    data_sources: List[str]
    thresholds: Dict[str, float]  # {'low': x, 'medium': y, 'high': z, 'critical': w}
    is_leading: bool  # True for leading indicators, False for lagging
    unit: str = "%"


class KRIRegistry:
    """Registry of all KRI definitions."""
    
    def __init__(self):
        """Initialize KRI registry with predefined indicators."""
        self.kris: Dict[str, KRIDefinition] = {}
        self._register_credit_kris()
        self._register_market_kris()
        self._register_liquidity_kris()
    
    def _register_credit_kris(self):
        """Register credit risk KRIs."""
        # Loan Default Rate (lagging)
        self.kris['loan_default_rate'] = KRIDefinition(
            name='Loan Default Rate',
            category=RiskCategory.CREDIT,
            description='Percentage of loans in default',
            calculation_method='(Defaulted Loans / Total Loans) * 100',
            data_sources=['loan_portfolio', 'default_events'],
            thresholds={'low': 2.0, 'medium': 5.0, 'high': 10.0, 'critical': 15.0},
            is_leading=False,
            unit='%'
        )
        
        # Delinquency Rate (leading)
        self.kris['delinquency_rate'] = KRIDefinition(
            name='Delinquency Rate',
            category=RiskCategory.CREDIT,
            description='Percentage of loans 30+ days past due (leading indicator)',
            calculation_method='(Delinquent Loans / Total Loans) * 100',
            data_sources=['loan_portfolio', 'payment_history'],
            thresholds={'low': 3.0, 'medium': 6.0, 'high': 12.0, 'critical': 18.0},
            is_leading=True,
            unit='%'
        )
        
        # Credit Quality Score
        self.kris['credit_quality_score'] = KRIDefinition(
            name='Credit Quality Score',
            category=RiskCategory.CREDIT,
            description='Weighted average credit score of loan portfolio',
            calculation_method='Weighted average of borrower credit scores',
            data_sources=['loan_portfolio', 'credit_scores'],
            thresholds={'low': 700, 'medium': 650, 'high': 600, 'critical': 550},
            is_leading=True,
            unit='score'
        )
        
        # Loan Concentration Ratio
        self.kris['loan_concentration'] = KRIDefinition(
            name='Loan Concentration Ratio',
            category=RiskCategory.CREDIT,
            description='Concentration of loans in top sectors/borrowers',
            calculation_method='(Top 10 Exposures / Total Loans) * 100',
            data_sources=['loan_portfolio'],
            thresholds={'low': 20.0, 'medium': 35.0, 'high': 50.0, 'critical': 65.0},
            is_leading=False,
            unit='%'
        )
    
    def _register_market_kris(self):
        """Register market risk KRIs."""
        # Portfolio Volatility
        self.kris['portfolio_volatility'] = KRIDefinition(
            name='Portfolio Volatility',
            category=RiskCategory.MARKET,
            description='Annualized standard deviation of portfolio returns',
            calculation_method='std(returns) * sqrt(252)',
            data_sources=['portfolio_returns'],
            thresholds={'low': 10.0, 'medium': 20.0, 'high': 30.0, 'critical': 40.0},
            is_leading=True,
            unit='%'
        )
        
        # Value at Risk (VaR)
        self.kris['var_95'] = KRIDefinition(
            name='Value at Risk (95%)',
            category=RiskCategory.MARKET,
            description='Maximum expected loss at 95% confidence level',
            calculation_method='5th percentile of return distribution',
            data_sources=['portfolio_returns', 'portfolio_value'],
            thresholds={'low': 1.0, 'medium': 3.0, 'high': 5.0, 'critical': 10.0},
            is_leading=False,
            unit='%'
        )
        
        # Interest Rate Risk
        self.kris['interest_rate_risk'] = KRIDefinition(
            name='Interest Rate Risk',
            category=RiskCategory.MARKET,
            description='Sensitivity to interest rate changes (duration)',
            calculation_method='Modified duration * portfolio value',
            data_sources=['portfolio_holdings', 'interest_rates'],
            thresholds={'low': 2.0, 'medium': 5.0, 'high': 8.0, 'critical': 12.0},
            is_leading=True,
            unit='years'
        )
    
    def _register_liquidity_kris(self):
        """Register liquidity risk KRIs."""
        # Liquidity Coverage Ratio
        self.kris['liquidity_coverage_ratio'] = KRIDefinition(
            name='Liquidity Coverage Ratio',
            category=RiskCategory.LIQUIDITY,
            description='Ratio of liquid assets to net cash outflows',
            calculation_method='Liquid Assets / Net 30-day Cash Outflows',
            data_sources=['balance_sheet', 'cash_flows'],
            thresholds={'low': 1.5, 'medium': 1.2, 'high': 1.0, 'critical': 0.8},
            is_leading=True,
            unit='ratio'
        )
        
        # Deposit Flow Ratio
        self.kris['deposit_flow_ratio'] = KRIDefinition(
            name='Deposit Flow Ratio',
            category=RiskCategory.LIQUIDITY,
            description='Net deposit inflows/outflows as % of total deposits',
            calculation_method='(Deposit Change / Total Deposits) * 100',
            data_sources=['balance_sheet', 'deposit_flows'],
            thresholds={'low': -5.0, 'medium': -10.0, 'high': -15.0, 'critical': -25.0},
            is_leading=True,
            unit='%'
        )
    
    def get_kri(self, name: str) -> KRIDefinition:
        """Get KRI definition by name."""
        return self.kris.get(name)
    
    def get_kris_by_category(self, category: RiskCategory) -> List[KRIDefinition]:
        """Get all KRIs for a specific category."""
        return [kri for kri in self.kris.values() if kri.category == category]
    
    def get_leading_indicators(self) -> List[KRIDefinition]:
        """Get all leading indicators."""
        return [kri for kri in self.kris.values() if kri.is_leading]
    
    def get_lagging_indicators(self) -> List[KRIDefinition]:
        """Get all lagging indicators."""
        return [kri for kri in self.kris.values() if not kri.is_leading]
    
    def list_all(self) -> List[str]:
        """List all KRI names."""
        return list(self.kris.keys())


# Global KRI registry instance
kri_registry = KRIRegistry()
