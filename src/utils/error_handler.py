"""
Error handling and resilience utilities for the risk forecasting system.

This module provides centralized error handling with:
- Retry logic with exponential backoff
- Cache fallback for data errors
- Graceful degradation for model failures
- Comprehensive error logging with context
"""
import time
import functools
from typing import Optional, Callable, Any, Dict, List, Type, Union
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logging_config import logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in the system."""
    DATA_FETCH = "data_fetch"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    SIMULATION = "simulation"
    VISUALIZATION = "visualization"
    SYSTEM = "system"


class RiskForecastingError(Exception):
    """Base exception for risk forecasting system."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()


class DataFetchError(RiskForecastingError):
    """Error during data fetching operations."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message,
            category=ErrorCategory.DATA_FETCH,
            severity=ErrorSeverity.HIGH,
            context=context
        )


class ModelError(RiskForecastingError):
    """Error during model operations."""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        context['model_name'] = model_name
        super().__init__(
            message,
            category=ErrorCategory.MODEL_TRAINING,
            severity=severity,
            context=context
        )


class ErrorHandler:
    """
    Centralized error handler for the risk forecasting system.
    
    Provides methods for handling different types of errors with appropriate
    recovery strategies including retry logic, cache fallback, and graceful degradation.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        cache_staleness_hours: int = 168  # 1 week
    ):
        """
        Initialize error handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds between retries
            cache_staleness_hours: Maximum age of stale cache to use as fallback
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.cache_staleness_hours = cache_staleness_hours
        self.error_history: List[RiskForecastingError] = []
        
    def handle_data_error(
        self,
        error: Exception,
        series_id: str,
        cache_path: Optional[Path] = None,
        use_stale_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Handle data fetching errors with cache fallback.
        
        Args:
            error: The exception that occurred
            series_id: Identifier of the data series
            cache_path: Path to cached data file
            use_stale_cache: Whether to use stale cache as fallback
            
        Returns:
            Cached data if available, None otherwise
        """
        context = {
            'series_id': series_id,
            'error_type': type(error).__name__,
            'error_message': str(error)
        }
        
        # Log the error
        logger.error(
            f"Data fetch error for {series_id}: {error}",
            extra={'context': context}
        )
        
        # Record error
        data_error = DataFetchError(
            f"Failed to fetch data for {series_id}: {error}",
            context=context
        )
        self.error_history.append(data_error)
        
        # Try to use stale cache as fallback
        if use_stale_cache and cache_path and cache_path.exists():
            cache_age = self._get_cache_age(cache_path)
            
            if cache_age < timedelta(hours=self.cache_staleness_hours):
                logger.warning(
                    f"Using stale cache for {series_id} "
                    f"(age: {cache_age.days} days, {cache_age.seconds // 3600} hours)"
                )
                
                try:
                    data = self._load_stale_cache(cache_path)
                    if data is not None and not data.empty:
                        logger.info(f"Successfully loaded stale cache for {series_id}")
                        return data
                except Exception as cache_error:
                    logger.error(f"Failed to load stale cache: {cache_error}")
            else:
                logger.error(
                    f"Cache too old for {series_id} "
                    f"(age: {cache_age.days} days, max: {self.cache_staleness_hours // 24} days)"
                )
        
        # No fallback available
        logger.error(f"No fallback data available for {series_id}")
        return None
    
    def handle_model_error(
        self,
        error: Exception,
        model_name: str,
        fallback_models: Optional[List[Any]] = None,
        data: Optional[pd.Series] = None,
        horizon: Optional[int] = None
    ) -> Optional[Any]:
        """
        Handle model errors with graceful degradation to simpler models.
        
        Args:
            error: The exception that occurred
            model_name: Name of the failed model
            fallback_models: List of simpler models to try
            data: Training data for fallback models
            horizon: Forecast horizon for fallback models
            
        Returns:
            Result from fallback model if successful, None otherwise
        """
        context = {
            'model_name': model_name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'has_fallback': fallback_models is not None and len(fallback_models) > 0
        }
        
        # Log the error
        logger.error(
            f"Model error in {model_name}: {error}",
            extra={'context': context}
        )
        
        # Record error
        model_error = ModelError(
            f"Model {model_name} failed: {error}",
            model_name=model_name,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        self.error_history.append(model_error)
        
        # Try fallback models
        if fallback_models and data is not None:
            logger.info(f"Attempting graceful degradation with {len(fallback_models)} fallback models")
            
            for fallback_model in fallback_models:
                try:
                    fallback_name = getattr(fallback_model, 'name', fallback_model.__class__.__name__)
                    logger.info(f"Trying fallback model: {fallback_name}")
                    
                    # Try to fit and forecast with fallback model
                    fallback_model.fit(data)
                    
                    if horizon is not None:
                        result = fallback_model.forecast(horizon)
                        logger.info(f"Successfully used fallback model: {fallback_name}")
                        return result
                    else:
                        logger.info(f"Successfully fitted fallback model: {fallback_name}")
                        return fallback_model
                    
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback model {fallback_name} also failed: {fallback_error}"
                    )
                    continue
            
            logger.error(f"All fallback models failed for {model_name}")
        
        return None
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        max_retries: Optional[int] = None,
        exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments for the function
            max_retries: Maximum retry attempts (uses instance default if None)
            exceptions: Tuple of exceptions to catch and retry
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result from successful function call
            
        Raises:
            Last exception if all retries fail
        """
        max_retries = max_retries or self.max_retries
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry successful on attempt {attempt + 1}")
                
                return result
                
            except exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_retries + 1} attempts failed. Last error: {e}"
                    )
        
        # All retries exhausted
        raise last_exception
    
    def with_error_handling(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        fallback_value: Any = None,
        raise_on_error: bool = False
    ):
        """
        Decorator for adding error handling to functions.
        
        Args:
            category: Error category
            severity: Error severity level
            fallback_value: Value to return on error if not raising
            raise_on_error: Whether to re-raise the exception
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'args': str(args)[:200],  # Truncate long args
                        'kwargs': str(kwargs)[:200],
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                    
                    # Log error with context
                    logger.error(
                        f"Error in {func.__name__}: {e}",
                        extra={'context': context},
                        exc_info=True
                    )
                    
                    # Record error
                    error = RiskForecastingError(
                        f"Error in {func.__name__}: {e}",
                        category=category,
                        severity=severity,
                        context=context
                    )
                    self.error_history.append(error)
                    
                    if raise_on_error:
                        raise
                    
                    return fallback_value
            
            return wrapper
        return decorator
    
    def _get_cache_age(self, cache_path: Path) -> timedelta:
        """Get age of cached file."""
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime
    
    def _load_stale_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Load data from stale cache file."""
        import json
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.set_index('date')
                return df[['value']]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse stale cache: {e}")
            return None
    
    def get_error_summary(
        self,
        since: Optional[datetime] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None
    ) -> Dict[str, Any]:
        """
        Get summary of errors that occurred.
        
        Args:
            since: Only include errors after this timestamp
            category: Filter by error category
            severity: Filter by error severity
            
        Returns:
            Dictionary with error statistics
        """
        filtered_errors = self.error_history
        
        # Apply filters
        if since:
            filtered_errors = [e for e in filtered_errors if e.timestamp >= since]
        
        if category:
            filtered_errors = [e for e in filtered_errors if e.category == category]
        
        if severity:
            filtered_errors = [e for e in filtered_errors if e.severity == severity]
        
        # Calculate statistics
        total_errors = len(filtered_errors)
        
        if total_errors == 0:
            return {
                'total_errors': 0,
                'by_category': {},
                'by_severity': {},
                'recent_errors': []
            }
        
        # Group by category
        by_category = {}
        for error in filtered_errors:
            cat = error.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        # Group by severity
        by_severity = {}
        for error in filtered_errors:
            sev = error.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        # Get recent errors
        recent_errors = sorted(
            filtered_errors,
            key=lambda e: e.timestamp,
            reverse=True
        )[:10]
        
        recent_error_info = [
            {
                'timestamp': e.timestamp.isoformat(),
                'category': e.category.value,
                'severity': e.severity.value,
                'message': e.message,
                'context': e.context
            }
            for e in recent_errors
        ]
        
        return {
            'total_errors': total_errors,
            'by_category': by_category,
            'by_severity': by_severity,
            'recent_errors': recent_error_info
        }
    
    def clear_error_history(self, before: Optional[datetime] = None):
        """
        Clear error history.
        
        Args:
            before: Only clear errors before this timestamp (clears all if None)
        """
        if before:
            self.error_history = [
                e for e in self.error_history
                if e.timestamp >= before
            ]
            logger.info(f"Cleared errors before {before}")
        else:
            count = len(self.error_history)
            self.error_history.clear()
            logger.info(f"Cleared all {count} errors from history")


# Global error handler instance
error_handler = ErrorHandler()


def retry_on_failure(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    base_delay: float = 1.0
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exceptions to catch and retry
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return error_handler.retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                exceptions=exceptions,
                **kwargs
            )
        return wrapper
    return decorator


def handle_errors(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    fallback_value: Any = None,
    raise_on_error: bool = False
):
    """
    Decorator for adding error handling to functions.
    
    Args:
        category: Error category
        severity: Error severity level
        fallback_value: Value to return on error if not raising
        raise_on_error: Whether to re-raise the exception
        
    Returns:
        Decorated function
    """
    return error_handler.with_error_handling(
        category=category,
        severity=severity,
        fallback_value=fallback_value,
        raise_on_error=raise_on_error
    )
