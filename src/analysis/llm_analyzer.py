"""
LLM-powered economic analysis generator.

This module uses LLM agents to generate dynamic, context-aware economic analysis
instead of hardcoded interpretations.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.models.batch_llm_client import get_batch_client

logger = logging.getLogger(__name__)


class EconomicAnalyzer:
    """Generate economic analysis using LLM reasoning."""

    def __init__(self):
        """Initialize the analyzer with LLM client."""
        self.llm_client = get_batch_client()

        # Economic context knowledge base
        self.indicator_contexts = {
            "unemployment": {
                "full_name": "Unemployment Rate",
                "unit": "percentage",
                "target": "~4% (natural rate)",
                "policy_tool": "Monetary and fiscal policy",
                "relationships": ["inflation (Phillips curve)", "GDP growth", "consumer spending"]
            },
            "inflation": {
                "full_name": "Consumer Price Index (CPI)",
                "unit": "year-over-year percentage change",
                "target": "~2% (Federal Reserve target)",
                "policy_tool": "Federal funds rate",
                "relationships": ["interest rates", "unemployment", "wage growth"]
            },
            "interest_rate": {
                "full_name": "Federal Funds Rate",
                "unit": "percentage",
                "target": "Varies with economic conditions",
                "policy_tool": "Open market operations",
                "relationships": ["inflation", "credit spreads", "economic growth"]
            },
            "credit_spread": {
                "full_name": "BAA-Treasury 10Y Credit Spread",
                "unit": "percentage points",
                "target": "~2% in normal conditions",
                "policy_tool": "Credit markets, liquidity provision",
                "relationships": ["financial stress", "default risk", "lending conditions"]
            }
        }

    def analyze_indicator(
        self,
        series_name: str,
        historical_data: pd.Series,
        forecast_data: Optional[pd.Series] = None,
        other_indicators: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis for an economic indicator.

        Args:
            series_name: Name of the economic indicator
            historical_data: Historical time series data
            forecast_data: Forecasted future values
            other_indicators: Current values of related indicators

        Returns:
            Dictionary containing:
            - current_interpretation: Analysis of current state
            - forecast_implications: What the forecast suggests
            - risk_factors: Identified risks (LLM-generated)
            - economic_context: Broader economic significance
            - confidence_assessment: LLM's confidence in the analysis
        """
        # Calculate statistical features
        stats = self._calculate_statistics(historical_data, forecast_data)

        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            series_name,
            stats,
            other_indicators
        )

        # Generate analysis using LLM
        try:
            analysis = self._generate_llm_analysis(prompt, series_name)
        except Exception as e:
            logger.error(f"LLM analysis failed for {series_name}: {e}")
            # Fallback to basic statistical analysis
            analysis = self._fallback_analysis(series_name, stats)

        return analysis

    def _calculate_statistics(
        self,
        historical: pd.Series,
        forecast: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Calculate key statistics for the time series."""
        recent_values = historical.iloc[-6:]  # Last 6 months

        stats = {
            "current_value": float(historical.iloc[-1]),
            "recent_mean": float(recent_values.mean()),
            "recent_trend": float(recent_values.pct_change().mean() * 100),
            "volatility": float(historical.pct_change().std() * 100),
            "3month_change": float((historical.iloc[-1] - historical.iloc[-4]) / historical.iloc[-4] * 100) if len(historical) >= 4 else 0,
            "12month_change": float((historical.iloc[-1] - historical.iloc[-13]) / historical.iloc[-13] * 100) if len(historical) >= 13 else 0,
            "historical_mean": float(historical.mean()),
            "historical_std": float(historical.std()),
        }

        if forecast is not None and len(forecast) > 0:
            stats.update({
                "forecast_mean": float(forecast.mean()),
                "forecast_trend": float(forecast.pct_change().mean() * 100),
                "forecast_direction": "increasing" if forecast.iloc[-1] > forecast.iloc[0] else "decreasing",
                "forecast_change": float((forecast.iloc[-1] - stats["current_value"]) / stats["current_value"] * 100)
            })

        return stats

    def _build_analysis_prompt(
        self,
        series_name: str,
        stats: Dict[str, float],
        other_indicators: Optional[Dict[str, float]]
    ) -> str:
        """Build the prompt for LLM analysis."""
        context = self.indicator_contexts.get(series_name, {})

        prompt = f"""You are an expert economic analyst. Analyze the following economic indicator:

**Indicator**: {context.get('full_name', series_name.title())}
**Current Date**: {datetime.now().strftime('%Y-%m-%d')}

**Current Statistics**:
- Current Value: {stats['current_value']:.2f}{context.get('unit', '')}
- 3-Month Change: {stats['3month_change']:.2f}%
- 12-Month Change: {stats['12month_change']:.2f}%
- Recent Trend: {stats['recent_trend']:.3f}% per month
- Volatility: {stats['volatility']:.2f}%

**Historical Context**:
- Historical Mean: {stats['historical_mean']:.2f}
- Historical Std Dev: {stats['historical_std']:.2f}
- Target: {context.get('target', 'N/A')}
"""

        if 'forecast_change' in stats:
            prompt += f"""
**Forecast**:
- Direction: {stats['forecast_direction']}
- Expected Change: {stats['forecast_change']:.2f}% over forecast horizon
- Forecast Trend: {stats['forecast_trend']:.3f}% per month
"""

        if other_indicators:
            prompt += "\n**Related Indicators**:\n"
            for name, value in other_indicators.items():
                prompt += f"- {name.title()}: {value:.2f}\n"

        if 'relationships' in context:
            prompt += f"\n**Key Relationships**: {', '.join(context['relationships'])}\n"

        prompt += """
Provide a comprehensive analysis addressing:

1. **Current Interpretation** (2-3 sentences):
   - What does the current level tell us about the economy?
   - Is it elevated, suppressed, or at target levels?
   - What recent dynamics are notable?

2. **Forecast Implications** (2-3 sentences):
   - What does the forecast suggest about future economic conditions?
   - What changes should policymakers/investors anticipate?
   - How might this affect other economic variables?

3. **Risk Factors** (list 3-5 specific risks):
   - What could cause unexpected changes?
   - What vulnerabilities exist?
   - What external shocks could impact this indicator?

4. **Economic Context** (1-2 sentences):
   - Why is this indicator important?
   - Who should pay attention to this?

5. **Confidence Assessment** (1 sentence):
   - How confident are you in this analysis given the data quality and current uncertainty?

Respond in JSON format:
{{
    "current_interpretation": "string",
    "forecast_implications": "string",
    "risk_factors": ["risk1", "risk2", ...],
    "economic_context": "string",
    "confidence_assessment": "string"
}}
"""

        return prompt

    def _generate_llm_analysis(
        self,
        prompt: str,
        series_name: str
    ) -> Dict[str, Any]:
        """Generate analysis using LLM."""
        import json
        import re

        # Use batch client for single prompt
        responses = self.llm_client.batch_inference([prompt])

        if not responses or len(responses) == 0:
            raise ValueError("No response from LLM")

        response_text = responses[0]

        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                # Response is not JSON formatted, parse manually
                analysis = self._parse_text_response(response_text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            analysis = self._parse_text_response(response_text)

        # Validate required fields
        required_fields = [
            "current_interpretation",
            "forecast_implications",
            "risk_factors",
            "economic_context"
        ]

        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")

        return analysis

    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse non-JSON LLM response."""
        analysis = {
            "current_interpretation": "",
            "forecast_implications": "",
            "risk_factors": [],
            "economic_context": "",
            "confidence_assessment": "Moderate confidence based on available data"
        }

        # Simple text parsing (can be improved)
        sections = text.split('\n\n')
        for section in sections:
            lower_section = section.lower()
            if 'current interpretation' in lower_section:
                # Extract text after the header
                lines = section.split('\n')
                analysis["current_interpretation"] = ' '.join(lines[1:]).strip()
            elif 'forecast implication' in lower_section:
                lines = section.split('\n')
                analysis["forecast_implications"] = ' '.join(lines[1:]).strip()
            elif 'risk factor' in lower_section:
                # Extract list items
                lines = section.split('\n')
                risks = [line.strip('- *•').strip() for line in lines[1:] if line.strip().startswith(('- ', '* ', '• ', '1', '2', '3', '4', '5'))]
                analysis["risk_factors"] = risks
            elif 'economic context' in lower_section:
                lines = section.split('\n')
                analysis["economic_context"] = ' '.join(lines[1:]).strip()
            elif 'confidence' in lower_section:
                lines = section.split('\n')
                analysis["confidence_assessment"] = ' '.join(lines[1:]).strip()

        return analysis

    def _fallback_analysis(
        self,
        series_name: str,
        stats: Dict[str, float]
    ) -> Dict[str, Any]:
        """Provide basic statistical analysis if LLM fails."""
        context = self.indicator_contexts.get(series_name, {})

        # Determine trend description
        if abs(stats['recent_trend']) < 0.1:
            trend = "stable"
        elif stats['recent_trend'] > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        # Basic interpretation
        current_interp = (
            f"The {context.get('full_name', series_name)} is currently at "
            f"{stats['current_value']:.2f}{context.get('unit', '')},"
            f"showing a {trend} trend with {stats['3month_change']:.1f}% change over the last 3 months. "
            f"Volatility is at {stats['volatility']:.1f}%."
        )

        # Basic forecast implications
        if 'forecast_direction' in stats:
            forecast_impl = (
                f"The forecast indicates a {stats['forecast_direction']} trend, "
                f"with an expected {abs(stats['forecast_change']):.1f}% change over the forecast horizon. "
                f"This suggests {'tightening' if stats['forecast_direction'] == 'increasing' else 'easing'} conditions ahead."
            )
        else:
            forecast_impl = "No forecast data available for forward-looking analysis."

        # Generic risk factors
        risk_factors = [
            f"Unexpected shifts in {context.get('policy_tool', 'economic policy')}",
            "External economic shocks",
            "Changes in market sentiment",
            "Structural economic changes"
        ]

        return {
            "current_interpretation": current_interp,
            "forecast_implications": forecast_impl,
            "risk_factors": risk_factors,
            "economic_context": context.get('target', 'Important economic indicator for policy decisions'),
            "confidence_assessment": "Statistical analysis only; LLM-enhanced analysis unavailable"
        }

    def generate_comparative_analysis(
        self,
        indicators: Dict[str, pd.Series],
        forecasts: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis across multiple indicators.

        This analyzes how indicators relate to each other and identifies
        potential conflicts or confirmations in the economic signals.
        """
        # Calculate current values
        current_values = {name: float(series.iloc[-1]) for name, series in indicators.items()}

        # Build comparative prompt
        prompt = self._build_comparative_prompt(current_values, forecasts)

        try:
            # Generate analysis
            responses = self.llm_client.batch_inference([prompt])
            analysis_text = responses[0] if responses else ""

            # Parse response
            import json
            import re
            json_match = re.search(r'\{[\s\S]*\}', analysis_text)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                analysis = {
                    "overall_assessment": analysis_text[:500],
                    "key_tensions": ["Unable to parse detailed analysis"],
                    "dominant_theme": "Economic transition period",
                    "policy_implications": "Monitor key indicators closely"
                }
        except Exception as e:
            logger.error(f"Comparative analysis failed: {e}")
            analysis = {
                "overall_assessment": "Multiple indicators showing mixed signals",
                "key_tensions": ["Data analysis in progress"],
                "dominant_theme": "Economic monitoring required",
                "policy_implications": "Maintain current policy stance"
            }

        return analysis

    def _build_comparative_prompt(
        self,
        current_values: Dict[str, float],
        forecasts: Optional[Dict[str, pd.Series]]
    ) -> str:
        """Build prompt for comparative economic analysis."""
        prompt = f"""You are an expert macroeconomic analyst. Analyze the current state of the economy based on these key indicators:

**Current Indicator Values** (as of {datetime.now().strftime('%Y-%m-%d')}):
"""
        for name, value in current_values.items():
            context = self.indicator_contexts.get(name, {})
            prompt += f"- {context.get('full_name', name.title())}: {value:.2f}{context.get('unit', '')}\n"

        if forecasts:
            prompt += "\n**Forecast Directions**:\n"
            for name, forecast in forecasts.items():
                if len(forecast) > 0:
                    direction = "increasing" if forecast.iloc[-1] > current_values.get(name, 0) else "decreasing"
                    change = ((forecast.iloc[-1] - current_values.get(name, 0)) / current_values.get(name, 1)) * 100
                    prompt += f"- {name.title()}: {direction} ({change:+.1f}% expected)\n"

        prompt += """
Provide a comprehensive assessment addressing:

1. **Overall Economic Assessment**: What do these indicators collectively suggest about the current and future state of the economy? (3-4 sentences)

2. **Key Tensions**: Are there any conflicting signals? (e.g., inflation rising but unemployment also rising) List 2-3 tensions.

3. **Dominant Theme**: What is the primary economic narrative emerging from this data? (1-2 sentences)

4. **Policy Implications**: What should policymakers focus on given these signals? (2-3 sentences)

Respond in JSON format:
{{
    "overall_assessment": "string",
    "key_tensions": ["tension1", "tension2", ...],
    "dominant_theme": "string",
    "policy_implications": "string"
}}
"""

        return prompt


# Global analyzer instance
_analyzer = None

def get_analyzer() -> EconomicAnalyzer:
    """Get or create global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = EconomicAnalyzer()
    return _analyzer
