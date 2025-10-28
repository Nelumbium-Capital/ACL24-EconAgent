"""
Basic functionality tests for EconAgent-Light.
Tests core components without requiring LLM services.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mesa_model.utils import (
    pareto_skill_distribution,
    compute_income_tax,
    update_wages_and_prices,
    taylor_rule_interest_rate,
    validate_economic_parameters,
    US_TAX_BRACKETS,
    US_TAX_RATES_2018
)

from llm_integration.nemotron_client import (
    normalize_decision_value,
    validate_economic_decision
)

class TestEconomicUtils:
    """Test economic utility functions."""
    
    def test_pareto_skill_distribution(self):
        """Test Pareto skill distribution generation."""
        skills = pareto_skill_distribution(100, 8.0, 950.0)
        
        assert len(skills) == 100
        assert all(skill >= 1.0 for skill in skills)
        assert all(skill <= 950.0 for skill in skills)
        assert np.std(skills) > 0  # Should have variation
    
    def test_compute_income_tax(self):
        """Test income tax calculation."""
        # Test zero income
        assert compute_income_tax(0, US_TAX_BRACKETS, US_TAX_RATES_2018) == 0.0
        
        # Test low income (first bracket)
        tax = compute_income_tax(50, US_TAX_BRACKETS, US_TAX_RATES_2018)
        expected = 50 * US_TAX_RATES_2018[0]  # Should be in first bracket
        assert abs(tax - expected) < 0.01
        
        # Test higher income (multiple brackets)
        tax = compute_income_tax(1000, US_TAX_BRACKETS, US_TAX_RATES_2018)
        assert tax > 0
        assert tax < 1000  # Should be less than total income
    
    def test_update_wages_and_prices(self):
        """Test wage and price update logic."""
        new_wage, new_price = update_wages_and_prices(
            current_wage=1.0,
            current_price=1.0,
            employment_rate=0.95,
            inventory_level=1000.0,
            max_wage_inflation=0.05,
            max_price_inflation=0.1
        )
        
        assert new_wage > 0
        assert new_price > 0
        assert abs(new_wage - 1.0) <= 0.05  # Within inflation bounds
        assert abs(new_price - 1.0) <= 0.1   # Within inflation bounds
    
    def test_taylor_rule_interest_rate(self):
        """Test Taylor rule interest rate calculation."""
        rate = taylor_rule_interest_rate(
            current_rate=0.02,
            inflation_rate=0.03,
            unemployment_rate=0.05,
            natural_rate=0.01,
            inflation_target=0.02,
            unemployment_target=0.04
        )
        
        assert 0.0 <= rate <= 0.20  # Should be within bounds
        assert isinstance(rate, float)
    
    def test_validate_economic_parameters(self):
        """Test parameter validation."""
        params = {
            'productivity': -1.0,  # Should be made positive
            'n_agents': 0,         # Should be made at least 1
            'skill_change': 1.5    # Should be clamped to [0,1]
        }
        
        validated = validate_economic_parameters(params)
        
        assert validated['productivity'] > 0
        assert validated['n_agents'] >= 1
        assert 0 <= validated['skill_change'] <= 1

class TestLLMIntegration:
    """Test LLM integration components."""
    
    def test_normalize_decision_value(self):
        """Test decision value normalization."""
        # Test normal values
        assert normalize_decision_value(0.5) == 0.5
        assert normalize_decision_value(0.33) == 0.32  # Rounded to 0.02 step
        
        # Test boundary values
        assert normalize_decision_value(-0.1) == 0.0   # Clamped to 0
        assert normalize_decision_value(1.5) == 1.0    # Clamped to 1
        
        # Test string input
        assert normalize_decision_value("0.7") == 0.7
        
        # Test invalid input
        assert normalize_decision_value("invalid") == 0.5  # Fallback
    
    def test_validate_economic_decision(self):
        """Test economic decision validation."""
        # Test valid JSON
        valid_json = '{"work": 0.8, "consumption": 0.6}'
        result = validate_economic_decision(valid_json)
        
        assert result["work"] == 0.8
        assert result["consumption"] == 0.6
        
        # Test invalid JSON
        invalid_json = "not json"
        result = validate_economic_decision(invalid_json)
        
        assert result["work"] == 0.2  # Fallback values
        assert result["consumption"] == 0.1
        
        # Test JSON with out-of-range values
        out_of_range = '{"work": 1.5, "consumption": -0.2}'
        result = validate_economic_decision(out_of_range)
        
        assert result["work"] == 1.0   # Clamped
        assert result["consumption"] == 0.0  # Clamped

class TestModelIntegration:
    """Test model integration without LLM services."""
    
    def test_model_initialization(self):
        """Test that model can be initialized without LLM."""
        from mesa_model import EconModel
        
        model = EconModel(
            n_agents=10,
            episode_length=12,
            random_seed=42,
            llm_client=None,  # No LLM client
            enable_lightagent=False
        )
        
        assert model.n_agents == 10
        assert model.episode_length == 12
        assert len(model.schedule.agents) == 10
        assert model.current_step == 0
    
    def test_agent_fallback_decisions(self):
        """Test that agents can make fallback decisions."""
        from mesa_model import EconModel
        
        model = EconModel(
            n_agents=5,
            episode_length=3,
            random_seed=42,
            llm_client=None,
            enable_lightagent=False
        )
        
        # Run a few steps
        for _ in range(3):
            model.step()
        
        # Check that simulation completed
        assert model.current_step == 3
        
        # Check that agents made decisions
        for agent in model.schedule.agents:
            assert hasattr(agent, 'last_work_decision')
            assert hasattr(agent, 'last_consumption_decision')
            assert 0 <= agent.last_work_decision <= 1
            assert 0 <= agent.last_consumption_decision <= 1

def test_imports():
    """Test that all main modules can be imported."""
    try:
        from mesa_model import EconModel, EconAgent
        from llm_integration import UnifiedLLMClient
        from lightagent_integration import LightAgentWrapper
        from viz import EconResultsAnalyzer
        
        # If we get here, imports worked
        assert True
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])