"""
EconAgent-Light source package.
Modernized implementation of ACL24-EconAgent with Mesa + LightAgent + local LLMs.
"""

from .mesa_model import EconModel, EconAgent
from .lightagent_integration import LightAgentWrapper
from .llm_integration import UnifiedLLMClient

__version__ = "0.1.0"

__all__ = [
    "EconModel",
    "EconAgent",
    "LightAgentWrapper", 
    "UnifiedLLMClient"
]