"""
Deep VAR (Vector AutoRegression) forecasting model using PyTorch.

Implements a neural network-based VAR model that captures non-linear
multivariate interactions between economic variables.
"""
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.utils.logging_config import logger


class VARDataset(Dataset):
    """Dataset for Deep VAR training."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input features (lagged variables) shape (n_samples, n_features)
            y: Target values shape (n_samples, n_variables)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DeepVARNetwork(nn.Module):
    """Neural network architecture for Deep VAR."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2
    ):
        """
        Initialize Deep VAR network.
        
        Args:
            input_dim: Input dimension (n_variables * lag_order)
            output_dim: Output dimension (n_variables)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super(DeepVARNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class DeepVARForecaster(BaseForecaster):
    """
    Deep learning-based Vector AutoRegression for multivariate forecasting.
    
    Captures non-linear interactions between economic variables using
    a multi-layer feedforward neural network.
    """
    
    def __init__(
        self,
        lag_order: int = 12,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
        name: str = 'DeepVAR'
    ):
        """
        Initialize Deep VAR forecaster.
        
        Args:
            lag_order: Number of lagged observations to use as features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability for regularization
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs to wait before early stopping
            device: Device to use ('cpu', 'cuda', or None for auto)
            name: Name identifier for this forecaster
        """
        super().__init__(name)
        self.lag_order = lag_order
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.n_variables = None
        self.training_history = []
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'DeepVARForecaster':
        """
        Fit Deep VAR model to multivariate time series data.
        
        Args:
            data: Historical multivariate time series as DataFrame
                  with DatetimeIndex and multiple columns
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.name} model to {len(data)} observations with {len(data.columns)} variables")
        
        # Validate input
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame for multivariate forecasting")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")
        
        if len(data.columns) < 2:
            raise ValueError("Deep VAR requires at least 2 variables")
        
        # Store training data
        self.training_data = data
        self.n_variables = len(data.columns)
        self.variable_names = data.columns.tolist()
        
        # Prepare training data
        X, y = self._prepare_training_data(data)
        
        # Split into train and validation
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create datasets and dataloaders
        train_dataset = VARDataset(X_train, y_train)
        val_dataset = VARDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Initialize model
        input_dim = self.n_variables * self.lag_order
        output_dim = self.n_variables
        
        self.model = DeepVARNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        logger.info(f"Model architecture: {self.model}")
        
        # Train model
        self._train_model(train_loader, val_loader)
        
        self.is_fitted = True
        
        # Store metadata
        self.training_metadata = {
            'n_variables': self.n_variables,
            'variable_names': self.variable_names,
            'lag_order': self.lag_order,
            'hidden_dims': self.hidden_dims,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'final_train_loss': self.training_history[-1]['train_loss'],
            'final_val_loss': self.training_history[-1]['val_loss'],
            'epochs_trained': len(self.training_history),
            'fit_date': datetime.now().isoformat()
        }
        
        logger.info(f"Model fitted successfully. Final validation loss: {self.training_metadata['final_val_loss']:.6f}")
        
        return self
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data with lagged features.
        
        Args:
            data: Multivariate time series DataFrame
            
        Returns:
            Tuple of (X, y) where X contains lagged features and y contains targets
        """
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        data_values = data.values
        data_scaled = self.scaler_X.fit_transform(data_values)
        
        # Create lagged features
        X, y = [], []
        
        for i in range(self.lag_order, len(data_scaled)):
            # Features: flatten last lag_order observations
            X.append(data_scaled[i-self.lag_order:i].flatten())
            # Target: next observation
            y.append(data_scaled[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit y scaler (for inverse transform later)
        self.scaler_y.fit(data_values)
        
        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
        
        return X, y
    
    def _train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Train the Deep VAR model with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for up to {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Store history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> Dict[str, ForecastResult]:
        """
        Generate multi-step forecasts for all variables.
        
        Args:
            horizon: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dictionary mapping variable names to ForecastResult objects
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        logger.info(f"Generating {horizon}-step forecast for {self.n_variables} variables")
        
        self.model.eval()
        
        # Get last lag_order observations
        history = self.training_data.values[-self.lag_order:]
        history_scaled = self.scaler_X.transform(history)
        
        # Recursive forecasting
        forecasts_scaled = []
        current_input = history_scaled.flatten()
        
        with torch.no_grad():
            for step in range(horizon):
                # Prepare input
                input_tensor = torch.FloatTensor(current_input).unsqueeze(0).to(self.device)
                
                # Predict next step
                pred = self.model(input_tensor)
                pred_np = pred.cpu().numpy()[0]
                forecasts_scaled.append(pred_np)
                
                # Update input for next prediction (shift window)
                current_input = np.roll(current_input, -self.n_variables)
                current_input[-self.n_variables:] = pred_np
        
        forecasts_scaled = np.array(forecasts_scaled)
        
        # Inverse transform to original scale
        forecasts = self.scaler_y.inverse_transform(forecasts_scaled)
        
        # Calculate prediction intervals using residuals
        residuals = self._calculate_residuals()
        
        # Create ForecastResult for each variable
        results = {}
        
        for i, var_name in enumerate(self.variable_names):
            point_forecast = forecasts[:, i]
            
            # Calculate intervals
            lower_bound, upper_bound = self.calculate_prediction_intervals(
                point_forecast,
                residuals[:, i],
                confidence_level
            )
            
            results[var_name] = ForecastResult(
                model_name=self.name,
                series_name=var_name,
                forecast_date=datetime.now(),
                horizon=horizon,
                point_forecast=point_forecast,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                metadata={
                    'lag_order': self.lag_order,
                    'n_variables': self.n_variables,
                    'hidden_dims': self.hidden_dims
                }
            )
        
        logger.info(f"Forecast generated successfully for all variables")
        
        return results
    
    def _calculate_residuals(self) -> np.ndarray:
        """
        Calculate model residuals on training data.
        
        Returns:
            Array of residuals shape (n_samples, n_variables)
        """
        # Prepare data
        X, y = self._prepare_training_data(self.training_data)
        
        # Get predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = torch.FloatTensor(X[i:i+self.batch_size]).to(self.device)
                pred = self.model(batch_X)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate residuals in original scale
        predictions_original = self.scaler_y.inverse_transform(predictions)
        actuals_original = self.scaler_y.inverse_transform(y)
        
        residuals = actuals_original - predictions_original
        
        return residuals
    
    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history as DataFrame.
        
        Returns:
            DataFrame with training and validation losses
        """
        if not self.training_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.training_history)
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'metadata': self.training_metadata,
            'variable_names': self.variable_names,
            'n_variables': self.n_variables,
            'lag_order': self.lag_order,
            'hidden_dims': self.hidden_dims
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        self.training_metadata = checkpoint['metadata']
        self.variable_names = checkpoint['variable_names']
        self.n_variables = checkpoint['n_variables']
        self.lag_order = checkpoint['lag_order']
        self.hidden_dims = checkpoint['hidden_dims']
        
        # Recreate model
        input_dim = self.n_variables * self.lag_order
        output_dim = self.n_variables
        
        self.model = DeepVARNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
