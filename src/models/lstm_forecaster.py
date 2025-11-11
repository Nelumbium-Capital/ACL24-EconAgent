"""
LSTM (Long Short-Term Memory) forecasting model using PyTorch.

Implements sequence-based time series forecasting with LSTM neural networks.
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


class TimeSeriesDataset(Dataset):
    """Dataset for LSTM training with sequences."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences shape (n_samples, lookback_window, n_features)
            targets: Target values shape (n_samples, n_features)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class LSTMNetwork(nn.Module):
    """LSTM neural network architecture."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2
    ):
        """
        Initialize LSTM network.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            output_dim: Number of output features
            dropout: Dropout probability
        """
        super(LSTMNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor shape (batch_size, seq_len, input_dim)
            hidden: Optional hidden state tuple (h_0, c_0)
            
        Returns:
            Tuple of (output, hidden_state)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Take last time step output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        output = self.fc(last_output)
        
        return output, hidden


class LSTMForecaster(BaseForecaster):
    """
    LSTM-based time series forecaster.
    
    Uses Long Short-Term Memory networks to capture temporal dependencies
    in time series data for forecasting.
    """
    
    def __init__(
        self,
        lookback_window: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        gradient_clip_value: float = 1.0,
        lr_scheduler_patience: int = 5,
        lr_scheduler_factor: float = 0.5,
        device: Optional[str] = None,
        name: str = 'LSTM'
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            lookback_window: Number of past time steps to use as input
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
            learning_rate: Initial learning rate for optimizer
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            validation_split: Fraction of data for validation
            early_stopping_patience: Epochs to wait before early stopping
            gradient_clip_value: Maximum gradient norm for clipping
            lr_scheduler_patience: Epochs to wait before reducing learning rate
            lr_scheduler_factor: Factor to reduce learning rate by
            device: Device to use ('cpu', 'cuda', or None for auto)
            name: Name identifier for this forecaster
        """
        super().__init__(name)
        self.lookback_window = lookback_window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip_value = gradient_clip_value
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.scaler = None
        self.input_dim = None
        self.output_dim = None
        self.training_history = []
        
    def fit(self, data: pd.Series, **kwargs) -> 'LSTMForecaster':
        """
        Fit LSTM model to time series data.
        
        Args:
            data: Historical time series as pandas Series with DatetimeIndex
            **kwargs: Additional parameters
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.name} model to {len(data)} observations")
        
        # Validate data
        data = self.validate_data(data)
        self.training_data = data
        
        # Check if we have enough data
        if len(data) < self.lookback_window + 1:
            raise ValueError(
                f"Data length ({len(data)}) must be greater than lookback_window ({self.lookback_window})"
            )
        
        # Prepare sequences
        sequences, targets = self._prepare_sequences(data)
        
        # Split into train and validation
        split_idx = int(len(sequences) * (1 - self.validation_split))
        seq_train, seq_val = sequences[:split_idx], sequences[split_idx:]
        tgt_train, tgt_val = targets[:split_idx], targets[split_idx:]
        
        logger.info(f"Training samples: {len(seq_train)}, Validation samples: {len(seq_val)}")
        
        # Create datasets and dataloaders
        train_dataset = TimeSeriesDataset(seq_train, tgt_train)
        val_dataset = TimeSeriesDataset(seq_val, tgt_val)
        
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
        self.input_dim = 1  # Univariate time series
        self.output_dim = 1
        
        self.model = LSTMNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        logger.info(f"Model architecture: {self.model}")
        
        # Train model
        self._train_model(train_loader, val_loader)
        
        self.is_fitted = True
        
        # Store metadata
        self.training_metadata = {
            'lookback_window': self.lookback_window,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'training_samples': len(seq_train),
            'validation_samples': len(seq_val),
            'final_train_loss': self.training_history[-1]['train_loss'],
            'final_val_loss': self.training_history[-1]['val_loss'],
            'epochs_trained': len(self.training_history),
            'fit_date': datetime.now().isoformat()
        }
        
        logger.info(f"Model fitted successfully. Final validation loss: {self.training_metadata['final_val_loss']:.6f}")
        
        return self
    
    def _prepare_sequences(
        self,
        data: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM training.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (sequences, targets)
            - sequences: shape (n_samples, lookback_window, 1)
            - targets: shape (n_samples, 1)
        """
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        data_values = data.values.reshape(-1, 1)
        data_scaled = self.scaler.fit_transform(data_values)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.lookback_window, len(data_scaled)):
            # Sequence: lookback_window previous values
            sequences.append(data_scaled[i-self.lookback_window:i])
            # Target: next value
            targets.append(data_scaled[i])
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        logger.info(f"Prepared sequences: shape {sequences.shape}, targets: shape {targets.shape}")
        
        return sequences, targets
    
    def _train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """
        Train the LSTM model with gradient clipping and learning rate scheduling.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for up to {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_seq, batch_tgt in train_loader:
                batch_seq = batch_seq.to(self.device)
                batch_tgt = batch_tgt.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs, _ = self.model(batch_seq)
                loss = criterion(outputs, batch_tgt)
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_value
                )
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_seq, batch_tgt in val_loader:
                    batch_seq = batch_seq.to(self.device)
                    batch_tgt = batch_tgt.to(self.device)
                    
                    outputs, _ = self.model(batch_seq)
                    loss = criterion(outputs, batch_tgt)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store history
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr
            })
            
            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} - "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                    f"LR: {current_lr:.6f}"
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
    ) -> ForecastResult:
        """
        Generate multi-step forecasts using recursive prediction.
        
        Args:
            horizon: Number of steps to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            ForecastResult with point forecasts and intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        logger.info(f"Generating {horizon}-step forecast")
        
        self.model.eval()
        
        # Get last lookback_window observations
        history = self.training_data.values[-self.lookback_window:].reshape(-1, 1)
        history_scaled = self.scaler.transform(history)
        
        # Recursive forecasting
        forecasts_scaled = []
        current_sequence = history_scaled.copy()
        
        with torch.no_grad():
            for step in range(horizon):
                # Prepare input
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Predict next step
                pred, _ = self.model(input_tensor)
                pred_np = pred.cpu().numpy()[0]
                forecasts_scaled.append(pred_np)
                
                # Update sequence (shift window and add prediction)
                current_sequence = np.vstack([current_sequence[1:], pred_np])
        
        forecasts_scaled = np.array(forecasts_scaled)
        
        # Inverse transform to original scale
        forecasts = self.scaler.inverse_transform(forecasts_scaled).flatten()
        
        # Calculate prediction intervals using residuals
        residuals = self._calculate_residuals()
        lower_bound, upper_bound = self.calculate_prediction_intervals(
            forecasts,
            residuals,
            confidence_level
        )
        
        # Create result
        result = ForecastResult(
            model_name=self.name,
            series_name=self.training_data.name if self.training_data.name else 'series',
            forecast_date=datetime.now(),
            horizon=horizon,
            point_forecast=forecasts,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence_level=confidence_level,
            metadata={
                'lookback_window': self.lookback_window,
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers
            }
        )
        
        logger.info(f"Forecast generated successfully")
        
        return result
    
    def _calculate_residuals(self) -> np.ndarray:
        """
        Calculate model residuals on training data.
        
        Returns:
            Array of residuals
        """
        # Prepare sequences
        sequences, targets = self._prepare_sequences(self.training_data)
        
        # Get predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch_seq = torch.FloatTensor(
                    sequences[i:i+self.batch_size]
                ).to(self.device)
                pred, _ = self.model(batch_seq)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Calculate residuals in original scale
        predictions_original = self.scaler.inverse_transform(predictions).flatten()
        actuals_original = self.scaler.inverse_transform(targets).flatten()
        
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
            'scaler': self.scaler,
            'metadata': self.training_metadata,
            'lookback_window': self.lookback_window,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.scaler = checkpoint['scaler']
        self.training_metadata = checkpoint['metadata']
        self.lookback_window = checkpoint['lookback_window']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        
        # Recreate model
        self.input_dim = 1
        self.output_dim = 1
        
        self.model = LSTMNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_dim=self.output_dim,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
