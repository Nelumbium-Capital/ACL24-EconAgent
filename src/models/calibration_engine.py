"""
Calibration and backtesting engine for time-series forecasting models.

Provides functionality for:
- Time-series cross-validation backtesting
- Performance metrics calculation (MAE, RMSE, MAPE, directional accuracy)
- Hyperparameter optimization using grid search
- Model comparison and selection
- Automated retraining with configurable triggers
- Model versioning and rollback capability
"""
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import itertools
import json
import pickle
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.utils.logging_config import logger


@dataclass
class BacktestResult:
    """Results from a backtesting run."""
    model_name: str
    fold_results: List[Dict[str, float]]
    average_metrics: Dict[str, float]
    best_fold: int
    worst_fold: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'fold_results': self.fold_results,
            'average_metrics': self.average_metrics,
            'best_fold': self.best_fold,
            'worst_fold': self.worst_fold,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ModelVersion:
    """Model version information for tracking and rollback."""
    version_id: str
    model_name: str
    model_path: Path
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_date: datetime
    data_hash: str
    is_active: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'performance_metrics': self.performance_metrics,
            'hyperparameters': self.hyperparameters,
            'training_date': self.training_date.isoformat(),
            'data_hash': self.data_hash,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        return cls(
            version_id=data['version_id'],
            model_name=data['model_name'],
            model_path=Path(data['model_path']),
            performance_metrics=data['performance_metrics'],
            hyperparameters=data['hyperparameters'],
            training_date=datetime.fromisoformat(data['training_date']),
            data_hash=data['data_hash'],
            is_active=data.get('is_active', False)
        )


class CalibrationEngine:
    """
    Engine for calibrating, backtesting, and managing forecasting models.
    
    Provides comprehensive model evaluation, hyperparameter optimization,
    and automated retraining capabilities.
    """
    
    def __init__(
        self,
        models: Optional[List[BaseForecaster]] = None,
        model_registry_path: Optional[Path] = None
    ):
        """
        Initialize calibration engine.
        
        Args:
            models: List of forecaster models to calibrate
            model_registry_path: Path to store model versions and metadata
        """
        self.models = models or []
        self.model_registry_path = model_registry_path or Path("models/registry")
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        
        self.performance_history: List[Dict[str, Any]] = []
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Calibration engine initialized with {len(self.models)} models")
    
    def backtest(
        self,
        data: pd.Series,
        models: Optional[List[BaseForecaster]] = None,
        n_splits: int = 5,
        horizon: int = 12,
        min_train_size: Optional[int] = None,
        expanding_window: bool = True
    ) -> Dict[str, BacktestResult]:
        """
        Perform time-series cross-validation backtesting.
        
        Args:
            data: Historical time series data
            models: Models to backtest (uses self.models if None)
            n_splits: Number of cross-validation folds
            horizon: Forecast horizon for each fold
            min_train_size: Minimum training set size (default: 70% of data)
            expanding_window: If True, use expanding window; if False, use rolling window
            
        Returns:
            Dictionary mapping model names to BacktestResult objects
        """
        models = models or self.models
        
        if not models:
            raise ValueError("No models provided for backtesting")
        
        if min_train_size is None:
            min_train_size = int(len(data) * 0.7)
        
        logger.info(
            f"Starting backtest with {len(models)} models, "
            f"{n_splits} folds, horizon={horizon}"
        )
        
        # Create time series cross-validator
        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=horizon,
            gap=0
        )
        
        results = {}
        
        for model in models:
            logger.info(f"Backtesting model: {model.name}")
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data)):
                # Skip if training set is too small
                if len(train_idx) < min_train_size:
                    logger.warning(
                        f"Fold {fold_idx+1}: Training set too small ({len(train_idx)} < {min_train_size}), skipping"
                    )
                    continue
                
                # Split data
                if expanding_window:
                    train_data = data.iloc[:train_idx[-1]+1]
                else:
                    train_data = data.iloc[train_idx]
                
                test_data = data.iloc[test_idx]
                
                logger.info(
                    f"Fold {fold_idx+1}/{n_splits}: "
                    f"Train size={len(train_data)}, Test size={len(test_data)}"
                )
                
                try:
                    # Fit model
                    model.fit(train_data)
                    
                    # Generate forecast
                    forecast_result = model.forecast(horizon=len(test_data))
                    forecasts = forecast_result.point_forecast
                    
                    # Ensure same length
                    min_len = min(len(forecasts), len(test_data))
                    forecasts = forecasts[:min_len]
                    actuals = test_data.values[:min_len]
                    
                    # Compute metrics
                    metrics = self._compute_metrics(forecasts, actuals)
                    metrics['fold'] = fold_idx + 1
                    metrics['train_size'] = len(train_data)
                    metrics['test_size'] = len(test_data)
                    
                    fold_results.append(metrics)
                    
                    logger.info(
                        f"Fold {fold_idx+1} metrics: "
                        f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                        f"MAPE={metrics['mape']:.2f}%"
                    )
                    
                except Exception as e:
                    logger.error(f"Fold {fold_idx+1} failed for {model.name}: {e}")
                    continue
            
            if not fold_results:
                logger.warning(f"No successful folds for {model.name}")
                continue
            
            # Calculate average metrics
            avg_metrics = self._average_metrics(fold_results)
            
            # Find best and worst folds
            rmse_values = [f['rmse'] for f in fold_results]
            best_fold = int(np.argmin(rmse_values))
            worst_fold = int(np.argmax(rmse_values))
            
            # Create backtest result
            backtest_result = BacktestResult(
                model_name=model.name,
                fold_results=fold_results,
                average_metrics=avg_metrics,
                best_fold=best_fold,
                worst_fold=worst_fold
            )
            
            results[model.name] = backtest_result
            
            logger.info(
                f"{model.name} average metrics: "
                f"MAE={avg_metrics['mae']:.4f}, RMSE={avg_metrics['rmse']:.4f}, "
                f"MAPE={avg_metrics['mape']:.2f}%, "
                f"Directional Accuracy={avg_metrics['directional_accuracy']:.2%}"
            )
        
        # Store in performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'results': {name: result.to_dict() for name, result in results.items()}
        })
        
        return results
    
    def _compute_metrics(
        self,
        forecasts: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute forecast accuracy metrics.
        
        Args:
            forecasts: Predicted values
            actuals: Actual values
            
        Returns:
            Dictionary of metrics
        """
        errors = actuals - forecasts
        
        # Mean Absolute Error
        mae = float(np.mean(np.abs(errors)))
        
        # Root Mean Squared Error
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        
        # Mean Absolute Percentage Error
        non_zero_mask = actuals != 0
        if non_zero_mask.any():
            mape = float(np.mean(np.abs(errors[non_zero_mask] / actuals[non_zero_mask])) * 100)
        else:
            mape = np.nan
        
        # Directional Accuracy
        if len(actuals) > 1:
            actual_direction = np.sign(np.diff(actuals))
            forecast_direction = np.sign(np.diff(forecasts))
            directional_accuracy = float(np.mean(actual_direction == forecast_direction))
        else:
            directional_accuracy = np.nan
        
        # Mean Error (bias)
        mean_error = float(np.mean(errors))
        
        # Standard deviation of errors
        std_error = float(np.std(errors))
        
        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else np.nan
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'mean_error': mean_error,
            'std_error': std_error,
            'r_squared': r_squared
        }
    
    def _average_metrics(
        self,
        fold_results: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate average metrics across folds.
        
        Args:
            fold_results: List of metric dictionaries from each fold
            
        Returns:
            Dictionary of averaged metrics
        """
        if not fold_results:
            return {}
        
        # Get all metric keys (excluding non-numeric fields)
        metric_keys = [
            k for k in fold_results[0].keys()
            if k not in ['fold', 'train_size', 'test_size'] and
            not pd.isna(fold_results[0][k])
        ]
        
        avg_metrics = {}
        for key in metric_keys:
            values = [f[key] for f in fold_results if not pd.isna(f.get(key))]
            if values:
                avg_metrics[key] = float(np.mean(values))
                avg_metrics[f'{key}_std'] = float(np.std(values))
        
        return avg_metrics

    
    def optimize_hyperparameters(
        self,
        model_class: type,
        data: pd.Series,
        param_grid: Dict[str, List[Any]],
        n_splits: int = 3,
        horizon: int = 12,
        metric: str = 'rmse',
        minimize: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize model hyperparameters using grid search with cross-validation.
        
        Args:
            model_class: Model class to optimize (must inherit from BaseForecaster)
            data: Training data
            param_grid: Dictionary mapping parameter names to lists of values to try
            n_splits: Number of CV folds for evaluation
            horizon: Forecast horizon
            metric: Metric to optimize ('mae', 'rmse', 'mape', 'directional_accuracy')
            minimize: If True, minimize metric; if False, maximize
            
        Returns:
            Tuple of (best_params, best_score)
        """
        logger.info(
            f"Starting hyperparameter optimization for {model_class.__name__} "
            f"with {len(list(itertools.product(*param_grid.values())))} combinations"
        )
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        best_score = float('inf') if minimize else float('-inf')
        best_params = None
        results = []
        
        for idx, param_values in enumerate(param_combinations):
            param_dict = dict(zip(param_names, param_values))
            
            logger.info(f"Testing combination {idx+1}/{len(param_combinations)}: {param_dict}")
            
            try:
                # Create model instance with these parameters
                model = model_class(**param_dict)
                
                # Evaluate with cross-validation
                backtest_results = self.backtest(
                    data=data,
                    models=[model],
                    n_splits=n_splits,
                    horizon=horizon
                )
                
                if model.name not in backtest_results:
                    logger.warning(f"No results for {param_dict}, skipping")
                    continue
                
                # Get average metric
                avg_metrics = backtest_results[model.name].average_metrics
                score = avg_metrics.get(metric)
                
                if score is None or pd.isna(score):
                    logger.warning(f"Invalid score for {param_dict}, skipping")
                    continue
                
                results.append({
                    'params': param_dict,
                    'score': score,
                    'metrics': avg_metrics
                })
                
                logger.info(f"Score: {metric}={score:.6f}")
                
                # Update best
                is_better = (score < best_score) if minimize else (score > best_score)
                if is_better:
                    best_score = score
                    best_params = param_dict
                    logger.info(f"New best score: {best_score:.6f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {param_dict}: {e}")
                continue
        
        if best_params is None:
            raise ValueError("No valid parameter combinations found")
        
        logger.info(
            f"Optimization complete. Best params: {best_params}, "
            f"Best {metric}: {best_score:.6f}"
        )
        
        # Store optimization results
        self.performance_history.append({
            'timestamp': datetime.now(),
            'type': 'hyperparameter_optimization',
            'model_class': model_class.__name__,
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        })
        
        return best_params, best_score
    
    def compare_models(
        self,
        backtest_results: Dict[str, BacktestResult],
        metric: str = 'rmse',
        minimize: bool = True
    ) -> pd.DataFrame:
        """
        Compare model performance and rank them.
        
        Args:
            backtest_results: Results from backtest() method
            metric: Metric to use for comparison
            minimize: If True, lower is better; if False, higher is better
            
        Returns:
            DataFrame with model comparison sorted by performance
        """
        comparison_data = []
        
        for model_name, result in backtest_results.items():
            avg_metrics = result.average_metrics
            
            comparison_data.append({
                'model': model_name,
                'mae': avg_metrics.get('mae', np.nan),
                'mae_std': avg_metrics.get('mae_std', np.nan),
                'rmse': avg_metrics.get('rmse', np.nan),
                'rmse_std': avg_metrics.get('rmse_std', np.nan),
                'mape': avg_metrics.get('mape', np.nan),
                'mape_std': avg_metrics.get('mape_std', np.nan),
                'directional_accuracy': avg_metrics.get('directional_accuracy', np.nan),
                'directional_accuracy_std': avg_metrics.get('directional_accuracy_std', np.nan),
                'r_squared': avg_metrics.get('r_squared', np.nan),
                'n_folds': len(result.fold_results)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by metric
        if metric in df.columns:
            df = df.sort_values(metric, ascending=minimize)
            df['rank'] = range(1, len(df) + 1)
        
        logger.info(f"Model comparison (sorted by {metric}):")
        logger.info(f"\n{df.to_string()}")
        
        return df
    
    def select_best_model(
        self,
        backtest_results: Dict[str, BacktestResult],
        metric: str = 'rmse',
        minimize: bool = True
    ) -> str:
        """
        Select the best performing model based on backtest results.
        
        Args:
            backtest_results: Results from backtest() method
            metric: Metric to use for selection
            minimize: If True, select model with lowest metric; if False, highest
            
        Returns:
            Name of the best model
        """
        comparison = self.compare_models(backtest_results, metric, minimize)
        
        if comparison.empty:
            raise ValueError("No models to compare")
        
        best_model = comparison.iloc[0]['model']
        best_score = comparison.iloc[0][metric]
        
        logger.info(f"Best model: {best_model} with {metric}={best_score:.6f}")
        
        return best_model
    
    def save_model_version(
        self,
        model: BaseForecaster,
        data: pd.Series,
        performance_metrics: Dict[str, float],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        Save a model version with metadata for tracking and rollback.
        
        Args:
            model: Fitted model to save
            data: Training data used
            performance_metrics: Performance metrics for this version
            hyperparameters: Model hyperparameters
            
        Returns:
            ModelVersion object
        """
        if not model.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{model.name}_{timestamp}"
        
        # Calculate data hash for tracking
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data).values
        ).hexdigest()[:16]
        
        # Create model directory
        model_dir = self.model_registry_path / model.name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / f"{version_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Saved model to {model_path}")
        
        # Create version object
        version = ModelVersion(
            version_id=version_id,
            model_name=model.name,
            model_path=model_path,
            performance_metrics=performance_metrics,
            hyperparameters=hyperparameters or {},
            training_date=datetime.now(),
            data_hash=data_hash,
            is_active=False
        )
        
        # Add to registry
        if model.name not in self.model_versions:
            self.model_versions[model.name] = []
        
        self.model_versions[model.name].append(version)
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Created model version: {version_id}")
        
        return version
    
    def load_model_version(
        self,
        version_id: str
    ) -> BaseForecaster:
        """
        Load a specific model version.
        
        Args:
            version_id: Version ID to load
            
        Returns:
            Loaded model
        """
        # Find version
        version = None
        for versions in self.model_versions.values():
            for v in versions:
                if v.version_id == version_id:
                    version = v
                    break
            if version:
                break
        
        if version is None:
            raise ValueError(f"Version {version_id} not found")
        
        # Load model
        with open(version.model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loaded model version: {version_id}")
        
        return model
    
    def activate_model_version(
        self,
        version_id: str
    ):
        """
        Set a model version as active (for production use).
        
        Args:
            version_id: Version ID to activate
        """
        # Find version
        version = None
        model_name = None
        
        for name, versions in self.model_versions.items():
            for v in versions:
                if v.version_id == version_id:
                    version = v
                    model_name = name
                    break
            if version:
                break
        
        if version is None:
            raise ValueError(f"Version {version_id} not found")
        
        # Deactivate all other versions for this model
        for v in self.model_versions[model_name]:
            v.is_active = False
        
        # Activate this version
        version.is_active = True
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Activated model version: {version_id}")
    
    def get_active_model(
        self,
        model_name: str
    ) -> Optional[BaseForecaster]:
        """
        Get the currently active model version.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Active model or None if no active version
        """
        if model_name not in self.model_versions:
            return None
        
        for version in self.model_versions[model_name]:
            if version.is_active:
                return self.load_model_version(version.version_id)
        
        return None
    
    def rollback_model(
        self,
        model_name: str,
        version_id: Optional[str] = None
    ):
        """
        Rollback to a previous model version.
        
        Args:
            model_name: Name of the model
            version_id: Specific version to rollback to (if None, uses previous version)
        """
        if model_name not in self.model_versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        versions = sorted(
            self.model_versions[model_name],
            key=lambda v: v.training_date,
            reverse=True
        )
        
        if version_id is None:
            # Find current active version
            current_idx = None
            for idx, v in enumerate(versions):
                if v.is_active:
                    current_idx = idx
                    break
            
            if current_idx is None or current_idx >= len(versions) - 1:
                raise ValueError("No previous version to rollback to")
            
            # Rollback to previous version
            version_id = versions[current_idx + 1].version_id
        
        # Activate the rollback version
        self.activate_model_version(version_id)
        
        logger.info(f"Rolled back {model_name} to version {version_id}")
    
    def _save_registry(self):
        """Save model registry to disk."""
        registry_file = self.model_registry_path / "registry.json"
        
        registry_data = {
            model_name: [v.to_dict() for v in versions]
            for model_name, versions in self.model_versions.items()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.debug(f"Saved model registry to {registry_file}")
    
    def _load_registry(self):
        """Load model registry from disk."""
        registry_file = self.model_registry_path / "registry.json"
        
        if not registry_file.exists():
            logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            self.model_versions = {
                model_name: [ModelVersion.from_dict(v) for v in versions]
                for model_name, versions in registry_data.items()
            }
            
            logger.info(f"Loaded model registry with {len(self.model_versions)} models")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.model_versions = {}
    
    def get_performance_history(self) -> pd.DataFrame:
        """
        Get performance history as DataFrame.
        
        Returns:
            DataFrame with historical performance data
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        # Flatten history for DataFrame
        records = []
        for entry in self.performance_history:
            if 'results' in entry:
                # Backtest results
                for model_name, result_dict in entry['results'].items():
                    records.append({
                        'timestamp': entry['timestamp'],
                        'type': 'backtest',
                        'model': model_name,
                        **result_dict['average_metrics']
                    })
            elif entry.get('type') == 'hyperparameter_optimization':
                # Optimization results
                records.append({
                    'timestamp': entry['timestamp'],
                    'type': 'optimization',
                    'model': entry['model_class'],
                    'best_score': entry['best_score']
                })
        
        return pd.DataFrame(records)
    
    def export_results(
        self,
        output_path: Path,
        format: str = 'json'
    ):
        """
        Export calibration results to file.
        
        Args:
            output_path: Path to save results
            format: Output format ('json', 'csv', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        
        elif format == 'csv':
            df = self.get_performance_history()
            df.to_csv(output_path, index=False)
        
        elif format == 'excel':
            df = self.get_performance_history()
            df.to_excel(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported results to {output_path}")



@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    trigger_type: str  # 'time', 'performance', 'data', 'manual'
    time_interval: Optional[timedelta] = None  # For time-based triggers
    performance_threshold: Optional[float] = None  # For performance-based triggers
    performance_metric: str = 'rmse'  # Metric to monitor
    performance_degradation_pct: float = 10.0  # Percentage degradation to trigger
    min_new_data_points: int = 10  # Minimum new data points to trigger
    validation_metric_threshold: Optional[float] = None  # Threshold for deployment
    auto_deploy: bool = False  # Automatically deploy if validation passes
    notification_callback: Optional[Callable] = None  # Callback for notifications


class AutoRetrainingEngine:
    """
    Automated retraining system with configurable triggers and validation.
    
    Monitors model performance and data changes, automatically retraining
    and deploying models when conditions are met.
    """
    
    def __init__(
        self,
        calibration_engine: CalibrationEngine,
        config: RetrainingConfig,
        log_path: Optional[Path] = None
    ):
        """
        Initialize auto-retraining engine.
        
        Args:
            calibration_engine: CalibrationEngine instance for model management
            config: Retraining configuration
            log_path: Path to store retraining logs
        """
        self.calibration_engine = calibration_engine
        self.config = config
        self.log_path = log_path or Path("logs/retraining")
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.retraining_history: List[Dict[str, Any]] = []
        self.last_training_time: Dict[str, datetime] = {}
        self.last_data_hash: Dict[str, str] = {}
        self.baseline_performance: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"Auto-retraining engine initialized with trigger: {config.trigger_type}")
    
    def should_retrain(
        self,
        model_name: str,
        current_data: pd.Series,
        current_performance: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Check if model should be retrained based on configured triggers.
        
        Args:
            model_name: Name of the model to check
            current_data: Current training data
            current_performance: Current model performance metrics
            
        Returns:
            Tuple of (should_retrain, reason)
        """
        reasons = []
        
        # Time-based trigger
        if self.config.trigger_type == 'time' and self.config.time_interval:
            last_time = self.last_training_time.get(model_name)
            if last_time is None:
                reasons.append("No previous training time recorded")
            elif datetime.now() - last_time >= self.config.time_interval:
                reasons.append(
                    f"Time interval exceeded: {datetime.now() - last_time} >= {self.config.time_interval}"
                )
        
        # Performance-based trigger
        if self.config.trigger_type == 'performance' and current_performance:
            baseline = self.baseline_performance.get(model_name, {})
            metric = self.config.performance_metric
            
            if metric in baseline and metric in current_performance:
                baseline_value = baseline[metric]
                current_value = current_performance[metric]
                
                # Calculate degradation percentage
                degradation_pct = ((current_value - baseline_value) / baseline_value) * 100
                
                if degradation_pct >= self.config.performance_degradation_pct:
                    reasons.append(
                        f"Performance degraded: {metric} increased by {degradation_pct:.2f}% "
                        f"(threshold: {self.config.performance_degradation_pct}%)"
                    )
        
        # Data-based trigger
        if self.config.trigger_type == 'data':
            # Calculate data hash
            current_hash = hashlib.md5(
                pd.util.hash_pandas_object(current_data).values
            ).hexdigest()[:16]
            
            last_hash = self.last_data_hash.get(model_name)
            
            if last_hash is None:
                reasons.append("No previous data hash recorded")
            elif current_hash != last_hash:
                # Check if enough new data points
                # This is a simplified check; in practice, you'd track actual new points
                reasons.append(f"Data has changed (hash: {last_hash} -> {current_hash})")
        
        # Manual trigger
        if self.config.trigger_type == 'manual':
            # Manual triggers are handled externally
            pass
        
        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining needed"
        
        return should_retrain, reason
    
    def auto_retrain(
        self,
        model: BaseForecaster,
        data: pd.Series,
        validation_data: Optional[pd.Series] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[ModelVersion]]:
        """
        Automatically retrain model with validation and deployment.
        
        Args:
            model: Model to retrain
            data: Training data
            validation_data: Optional validation data for pre-deployment testing
            hyperparameters: Optional hyperparameters to use
            
        Returns:
            Tuple of (success, model_version)
        """
        model_name = model.name
        
        logger.info(f"Starting auto-retrain for {model_name}")
        
        # Check if retraining is needed
        current_performance = None
        if model.is_fitted:
            try:
                # Quick validation on recent data
                recent_data = data.iloc[-50:]
                forecast = model.forecast(horizon=10)
                actuals = recent_data.values[-10:]
                current_performance = self.calibration_engine._compute_metrics(
                    forecast.point_forecast[:len(actuals)],
                    actuals
                )
            except Exception as e:
                logger.warning(f"Could not compute current performance: {e}")
        
        should_retrain, reason = self.should_retrain(
            model_name,
            data,
            current_performance
        )
        
        if not should_retrain:
            logger.info(f"Retraining not needed for {model_name}: {reason}")
            return False, None
        
        logger.info(f"Retraining triggered for {model_name}: {reason}")
        
        # Record retraining event
        retraining_event = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'trigger_reason': reason,
            'data_size': len(data),
            'status': 'started'
        }
        
        try:
            # Retrain model
            logger.info(f"Retraining {model_name} on {len(data)} data points")
            model.fit(data)
            
            # Validate retrained model
            if validation_data is not None:
                logger.info("Validating retrained model")
                validation_passed, validation_metrics = self._validate_retrained_model(
                    model,
                    validation_data
                )
                
                retraining_event['validation_metrics'] = validation_metrics
                retraining_event['validation_passed'] = validation_passed
                
                if not validation_passed:
                    logger.warning(
                        f"Retrained model failed validation: "
                        f"{self.config.performance_metric}={validation_metrics[self.config.performance_metric]:.6f}"
                    )
                    retraining_event['status'] = 'failed_validation'
                    self.retraining_history.append(retraining_event)
                    self._save_retraining_log(retraining_event)
                    return False, None
            else:
                validation_metrics = {}
                validation_passed = True
            
            # Save model version
            logger.info("Saving retrained model version")
            model_version = self.calibration_engine.save_model_version(
                model=model,
                data=data,
                performance_metrics=validation_metrics,
                hyperparameters=hyperparameters
            )
            
            retraining_event['version_id'] = model_version.version_id
            retraining_event['status'] = 'completed'
            
            # Auto-deploy if configured
            if self.config.auto_deploy and validation_passed:
                logger.info(f"Auto-deploying {model_version.version_id}")
                self.calibration_engine.activate_model_version(model_version.version_id)
                retraining_event['deployed'] = True
            else:
                retraining_event['deployed'] = False
            
            # Update tracking
            self.last_training_time[model_name] = datetime.now()
            self.last_data_hash[model_name] = hashlib.md5(
                pd.util.hash_pandas_object(data).values
            ).hexdigest()[:16]
            
            if validation_metrics:
                self.baseline_performance[model_name] = validation_metrics
            
            # Save log
            self.retraining_history.append(retraining_event)
            self._save_retraining_log(retraining_event)
            
            # Send notification
            if self.config.notification_callback:
                self.config.notification_callback(retraining_event)
            
            logger.info(
                f"Auto-retrain completed successfully for {model_name}. "
                f"Version: {model_version.version_id}"
            )
            
            return True, model_version
            
        except Exception as e:
            logger.error(f"Auto-retrain failed for {model_name}: {e}")
            retraining_event['status'] = 'failed'
            retraining_event['error'] = str(e)
            self.retraining_history.append(retraining_event)
            self._save_retraining_log(retraining_event)
            return False, None
    
    def _validate_retrained_model(
        self,
        model: BaseForecaster,
        validation_data: pd.Series,
        horizon: int = 12
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Validate retrained model before deployment.
        
        Args:
            model: Retrained model
            validation_data: Validation data
            horizon: Forecast horizon for validation
            
        Returns:
            Tuple of (passed, metrics)
        """
        try:
            # Split validation data
            train_size = len(validation_data) - horizon
            if train_size < 10:
                logger.warning("Insufficient validation data")
                return True, {}  # Pass by default if not enough data
            
            train_val = validation_data.iloc[:train_size]
            test_val = validation_data.iloc[train_size:]
            
            # Generate forecast
            model.fit(train_val)
            forecast = model.forecast(horizon=len(test_val))
            
            # Compute metrics
            metrics = self.calibration_engine._compute_metrics(
                forecast.point_forecast[:len(test_val)],
                test_val.values
            )
            
            # Check threshold
            if self.config.validation_metric_threshold is not None:
                metric_value = metrics.get(self.config.performance_metric)
                if metric_value is None:
                    return True, metrics
                
                passed = metric_value <= self.config.validation_metric_threshold
                return passed, metrics
            
            return True, metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False, {}
    
    def _save_retraining_log(self, event: Dict[str, Any]):
        """Save retraining event to log file."""
        log_file = self.log_path / f"retraining_{datetime.now().strftime('%Y%m')}.jsonl"
        
        with open(log_file, 'a') as f:
            json.dump(event, f, default=str)
            f.write('\n')
    
    def get_retraining_history(
        self,
        model_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get retraining history as DataFrame.
        
        Args:
            model_name: Filter by model name
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            DataFrame with retraining history
        """
        if not self.retraining_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.retraining_history)
        
        # Apply filters
        if model_name:
            df = df[df['model_name'] == model_name]
        
        if start_date:
            df = df[df['timestamp'] >= start_date]
        
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        return df
    
    def get_retraining_stats(self) -> Dict[str, Any]:
        """
        Get statistics about retraining activity.
        
        Returns:
            Dictionary with retraining statistics
        """
        if not self.retraining_history:
            return {}
        
        df = pd.DataFrame(self.retraining_history)
        
        stats = {
            'total_retrainings': len(df),
            'successful': len(df[df['status'] == 'completed']),
            'failed': len(df[df['status'].isin(['failed', 'failed_validation'])]),
            'deployed': len(df[df.get('deployed', False) == True]),
            'models_retrained': df['model_name'].nunique(),
            'avg_data_size': df['data_size'].mean() if 'data_size' in df else None,
            'last_retraining': df['timestamp'].max() if 'timestamp' in df else None
        }
        
        # Per-model stats
        stats['by_model'] = {}
        for model_name in df['model_name'].unique():
            model_df = df[df['model_name'] == model_name]
            stats['by_model'][model_name] = {
                'total': len(model_df),
                'successful': len(model_df[model_df['status'] == 'completed']),
                'last_retraining': model_df['timestamp'].max()
            }
        
        return stats
    
    def schedule_retraining(
        self,
        models: List[BaseForecaster],
        data_provider: Callable[[], pd.Series],
        check_interval: timedelta = timedelta(hours=1)
    ):
        """
        Schedule periodic retraining checks.
        
        Note: This is a simple implementation. For production, use a proper
        scheduler like APScheduler or Celery.
        
        Args:
            models: List of models to monitor
            data_provider: Callable that returns current training data
            check_interval: How often to check for retraining needs
        """
        logger.info(
            f"Scheduling retraining checks every {check_interval} "
            f"for {len(models)} models"
        )
        
        import time
        
        while True:
            try:
                # Get current data
                current_data = data_provider()
                
                # Check each model
                for model in models:
                    try:
                        self.auto_retrain(model, current_data)
                    except Exception as e:
                        logger.error(f"Retraining check failed for {model.name}: {e}")
                
                # Wait for next check
                time.sleep(check_interval.total_seconds())
                
            except KeyboardInterrupt:
                logger.info("Retraining scheduler stopped")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait a minute before retrying
