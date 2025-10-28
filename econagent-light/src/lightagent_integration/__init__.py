"""
LightAgent integration module for EconAgent-Light.
Provides LightAgent framework integration with economic agents.
"""

from .light_client import (
    LightAgentWrapper,
    AgentProfile,
    EnvironmentSnapshot,
    EconomicMemory
)

__all__ = [
    "LightAgentWrapper",
    "AgentProfile", 
    "EnvironmentSnapshot",
    "EconomicMemory"
]