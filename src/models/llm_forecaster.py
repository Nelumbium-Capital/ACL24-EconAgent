"""
LLM-based time series forecasting using Nemotron.
Inspired by foundation models like Chronos and TimesFM.
"""
import json
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import requests

from config import settings
from src.utils.logging_config import logger


class NemotronTimeSeriesForecaster:
    """
    Time series forecaster using Nemotron LLM.
    Uses prompt engineering to leverage LLM's pattern recognition for forecasting.
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/llama-3.1-nemotron-70b-instruct",
        use_ollama_fallback: bool = True
    ):
        """
        Initialize Nemotron forecaster.
        
        Args:
            model_name: Model identifier
            use_ollama_fallback: Whether to fall back to Ollama if Nemotron unavailable
        """
        self.model_name = model_name
        self.use_ollama_fallback = use_ollama_fallback
        self.nemotron_url = settings.nemotron_url
        self.ollama_url = settings.ollama_url
        
        logger.info(f"Initialized NemotronTimeSeriesForecaster with model: {model_name}")
    
    def _call_llm(
        self, 
        messages: List[dict], 
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """
        Call LLM API with fallback logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Try Nemotron first
        try:
            response = requests.post(
                f"{self.nemotron_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.warning(f"Nemotron returned status {response.status_code}")
        
        except Exception as e:
            logger.warning(f"Nemotron call failed: {e}")
        
        # Fallback to Ollama
        if self.use_ollama_fallback:
            try:
                logger.info("Falling back to Ollama")
                response = requests.post(
                    f"{self.ollama_url}/chat/completions",
                    json={
                        "model": "llama3.1",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
            
            except Exception as e:
                logger.error(f"Ollama fallback failed: {e}")
        
        raise RuntimeError("Both Nemotron and Ollama failed")
    
    def _prepare_time_series_prompt(
        self,
        series: np.ndarray,
        series_name: str,
        horizon: int,
        context_info: Optional[str] = None
    ) -> str:
        """
        Prepare prompt for time series forecasting.
        
        Args:
            series: Historical time series data
            series_name: Name of the series
            horizon: Number of steps to forecast
            context_info: Additional context about the series
            
        Returns:
            Formatted prompt string
        """
        # Format historical data
        recent_values = series[-20:]  # Last 20 observations
        values_str = ", ".join([f"{v:.4f}" for v in recent_values])
        
        # Calculate basic statistics
        mean = np.mean(series)
        std = np.std(series)
        trend = "increasing" if series[-1] > series[-10] else "decreasing"
        
        # Build prompt
        prompt = f"""You are an expert time series forecaster. Analyze the following economic time series and provide forecasts.

Series: {series_name}
Historical Data (most recent 20 observations): [{values_str}]

Statistics:
- Mean: {mean:.4f}
- Std Dev: {std:.4f}
- Recent Trend: {trend}
- Latest Value: {series[-1]:.4f}
"""
        
        if context_info:
            prompt += f"\nContext: {context_info}\n"
        
        prompt += f"""
Task: Forecast the next {horizon} values for this time series.

Instructions:
1. Analyze the patterns, trends, and seasonality in the historical data
2. Consider the economic context and recent trends
3. Provide {horizon} point forecasts
4. Return ONLY a JSON object with this exact format:
{{
    "forecasts": [value1, value2, ..., value{horizon}],
    "confidence": "high/medium/low",
    "reasoning": "brief explanation"
}}

Respond with valid JSON only, no additional text."""
        
        return prompt
    
    def forecast(
        self,
        series: np.ndarray,
        horizon: int,
        series_name: str = "time_series",
        context_info: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Generate forecasts using LLM.
        
        Args:
            series: Historical time series data
            horizon: Number of steps to forecast
            series_name: Name of the series
            context_info: Additional context
            
        Returns:
            Tuple of (forecasts array, reasoning string)
        """
        logger.info(f"Generating LLM forecast for {series_name}, horizon={horizon}")
        
        # Prepare prompt
        prompt = self._prepare_time_series_prompt(
            series, series_name, horizon, context_info
        )
        
        # Call LLM
        messages = [
            {
                "role": "system",
                "content": "You are an expert quantitative analyst specializing in time series forecasting and economic analysis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = self._call_llm(messages, temperature=0.3)
            
            # Parse JSON response
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            result = json.loads(json_str)
            
            forecasts = np.array(result['forecasts'])
            reasoning = result.get('reasoning', 'No reasoning provided')
            
            logger.info(f"LLM forecast complete: {len(forecasts)} values, confidence={result.get('confidence')}")
            
            return forecasts, reasoning
        
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: simple naive forecast
            logger.warning("Using naive forecast as fallback")
            naive_forecast = np.full(horizon, series[-1])
            return naive_forecast, "Fallback: naive forecast (last value repeated)"
    
    def forecast_with_uncertainty(
        self,
        series: np.ndarray,
        horizon: int,
        series_name: str = "time_series",
        n_samples: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts with uncertainty estimates using multiple LLM calls.
        
        Args:
            series: Historical time series data
            horizon: Number of steps to forecast
            series_name: Name of the series
            n_samples: Number of forecast samples to generate
            
        Returns:
            Tuple of (mean forecasts, lower bound, upper bound)
        """
        logger.info(f"Generating {n_samples} forecast samples for uncertainty estimation")
        
        all_forecasts = []
        
        for i in range(n_samples):
            try:
                forecasts, _ = self.forecast(series, horizon, series_name)
                all_forecasts.append(forecasts)
            except Exception as e:
                logger.warning(f"Sample {i+1} failed: {e}")
        
        if not all_forecasts:
            # Fallback
            naive = np.full(horizon, series[-1])
            return naive, naive * 0.95, naive * 1.05
        
        # Calculate statistics
        all_forecasts = np.array(all_forecasts)
        mean_forecast = np.mean(all_forecasts, axis=0)
        lower_bound = np.percentile(all_forecasts, 5, axis=0)
        upper_bound = np.percentile(all_forecasts, 95, axis=0)
        
        return mean_forecast, lower_bound, upper_bound


class LLMEnsembleForecaster:
    """
    Ensemble forecaster that combines LLM with traditional methods.
    """
    
    def __init__(self):
        """Initialize ensemble forecaster."""
        self.llm_forecaster = NemotronTimeSeriesForecaster()
        logger.info("Initialized LLM Ensemble Forecaster")
    
    def forecast(
        self,
        series: np.ndarray,
        horizon: int,
        series_name: str = "time_series",
        use_llm: bool = True
    ) -> dict:
        """
        Generate ensemble forecast.
        
        Args:
            series: Historical time series data
            horizon: Number of steps to forecast
            series_name: Name of the series
            use_llm: Whether to include LLM forecast
            
        Returns:
            Dictionary with forecasts and metadata
        """
        results = {
            'series_name': series_name,
            'horizon': horizon,
            'forecasts': {},
            'ensemble': None
        }
        
        # Naive forecast (baseline)
        naive_forecast = np.full(horizon, series[-1])
        results['forecasts']['naive'] = naive_forecast
        
        # Simple trend forecast
        if len(series) >= 2:
            trend = series[-1] - series[-2]
            trend_forecast = np.array([series[-1] + trend * (i+1) for i in range(horizon)])
            results['forecasts']['trend'] = trend_forecast
        
        # LLM forecast
        if use_llm:
            try:
                llm_forecast, reasoning = self.llm_forecaster.forecast(
                    series, horizon, series_name
                )
                results['forecasts']['llm'] = llm_forecast
                results['llm_reasoning'] = reasoning
            except Exception as e:
                logger.warning(f"LLM forecast failed: {e}")
        
        # Ensemble (simple average)
        forecast_arrays = list(results['forecasts'].values())
        if forecast_arrays:
            results['ensemble'] = np.mean(forecast_arrays, axis=0)
        
        return results
