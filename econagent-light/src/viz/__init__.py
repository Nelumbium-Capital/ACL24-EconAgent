"""
Visualization module for EconAgent-Light.
Provides plotting and analysis tools for simulation results.
"""

from .plot_results import EconResultsAnalyzer, create_analysis_report

__all__ = [
    "EconResultsAnalyzer",
    "create_analysis_report"
]