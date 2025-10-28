"""
Configuration module for EconAgent-Light system.
Migrates original config.yaml parameters to Python configuration.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class EconomicConfig:
    """Economic simulation parameters from original ACL24-EconAgent."""
    
    # Simulation parameters
    n_agents: int = 100
    episode_length: int = 240  # 20 years * 12 months
    random_seed: Optional[int] = None
    
    # Economic parameters (from original config.yaml)
    productivity: float = 1.0
    skill_change: float = 0.02
    price_change: float = 0.02
    max_price_inflation: float = 0.1
    max_wage_inflation: float = 0.05
    
    # Labor parameters
    pareto_param: float = 8.0  # For skill distribution
    payment_max_skill_multiplier: float = 950.0
    labor_step: int = 168  # Hours per month
    num_labor_hours: int = 168
    
    # Tax parameters
    tax_model: str = "us-federal-single-filer-2018-scaled"
    usd_scaling: int = 12  # Monthly scaling
    bracket_spacing: str = "us-federal"
    
    # Consumption parameters
    consumption_rate_step: float = 0.02
    
    # Saving parameters
    saving_rate: float = 0.00  # Base interest rate
    
    # Agent reward parameters
    agent_reward_type: str = "isoelastic_coin_minus_labor"
    isoelastic_etas: List[float] = None
    labor_exponent: float = 2.0
    labor_cost: float = 1.0
    
    # Economic dynamics
    enable_skill_change: bool = True
    enable_price_change: bool = True
    mixing_weight_gini_vs_coin: float = 0.0
    
    def __post_init__(self):
        if self.isoelastic_etas is None:
            self.isoelastic_etas = [0.5, 0.5]

@dataclass 
class LLMConfig:
    """LLM service configuration - supports both local and NVIDIA API."""
    
    # NVIDIA API configuration
    nvidia_api_key: str = "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"
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "nvidia/nemotron-4-340b-instruct"
    
    # Local Nemotron configuration (fallback)
    nemotron_base_url: str = "http://localhost:8000/v1"
    nemotron_model: str = "nvidia-nemotron-nano-9b-v2"
    
    # Ollama fallback configuration  
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama2:7b-chat"
    
    # Request parameters
    temperature: float = 0.2
    max_tokens: int = 512
    timeout: int = 30
    max_retries: int = 3
    
    # Performance parameters
    batch_size: int = 8
    parallel_threads: int = 8
    enable_caching: bool = True
    cache_size: int = 1000

@dataclass
class LightAgentConfig:
    """LightAgent framework configuration."""
    
    # Memory configuration
    memory_window: int = 1  # Months to retain
    enable_memory: bool = True
    
    # Tree of Thought configuration
    enable_tot: bool = True
    tot_candidates: int = 3
    
    # Reflection configuration
    reflection_frequency: int = 3  # Every 3 months
    
    # Tools configuration
    enable_tools: bool = True
    adaptive_tools: bool = True

@dataclass
class SystemConfig:
    """Overall system configuration."""
    
    economic: EconomicConfig
    llm: LLMConfig
    lightagent: LightAgentConfig
    
    # Logging and output
    log_level: str = "INFO"
    output_dir: str = "./results"
    save_frequency: int = 6  # Save every 6 months
    
    # Performance monitoring
    enable_profiling: bool = False
    monitor_resources: bool = True

# Default configuration instance
DEFAULT_CONFIG = SystemConfig(
    economic=EconomicConfig(),
    llm=LLMConfig(),
    lightagent=LightAgentConfig()
)

# Original tax brackets from simulate_utils.py
US_TAX_BRACKETS = [0, 97, 394.75, 842, 1607.25, 2041, 5103]  # Monthly brackets
US_TAX_BRACKETS_SCALED = [b * 100 / 12 for b in US_TAX_BRACKETS]

def load_config_from_env() -> SystemConfig:
    """Load configuration from environment variables."""
    config = DEFAULT_CONFIG
    
    # Override with environment variables if present
    if os.getenv("ECONAGENT_AGENTS"):
        config.economic.n_agents = int(os.getenv("ECONAGENT_AGENTS"))
    
    if os.getenv("ECONAGENT_YEARS"):
        years = int(os.getenv("ECONAGENT_YEARS"))
        config.economic.episode_length = years * 12
    
    if os.getenv("ECONAGENT_SEED"):
        config.economic.random_seed = int(os.getenv("ECONAGENT_SEED"))
    
    if os.getenv("NEMOTRON_URL"):
        config.llm.nemotron_base_url = os.getenv("NEMOTRON_URL")
    
    if os.getenv("OLLAMA_URL"):
        config.llm.ollama_base_url = os.getenv("OLLAMA_URL")
    
    return config