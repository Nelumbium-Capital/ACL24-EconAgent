"""
Kumo Graph Transformer for time series forecasting.

Implements a Graph Transformer architecture that models relationships between
economic variables as a learned graph structure. Based on the approach from
Kumo's time series forecasting methodology.

Key features:
- Automatic graph structure learning from data
- Multi-head attention for capturing temporal dependencies
- Node embeddings for economic indicators
- Multi-horizon forecasting support
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.models.base_forecaster import BaseForecaster, ForecastResult
from src.utils.logging_config import logger


class TimeSeriesGraphDataset(Dataset):
    """Dataset for graph-based time series forecasting."""

    def __init__(
        self,
        data: pd.DataFrame,
        lookback: int = 12,
        horizon: int = 6,
        normalize: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data: DataFrame with time series for multiple variables (shape: time x variables)
            lookback: Number of historical time steps to use
            horizon: Forecast horizon
            normalize: Whether to normalize the data
        """
        self.data = data.values
        self.lookback = lookback
        self.horizon = horizon
        self.n_vars = data.shape[1]
        self.var_names = list(data.columns)

        # Normalize data
        if normalize:
            self.mean = np.mean(self.data, axis=0, keepdims=True)
            self.std = np.std(self.data, axis=0, keepdims=True) + 1e-8
            self.normalized_data = (self.data - self.mean) / self.std
        else:
            self.normalized_data = self.data
            self.mean = np.zeros((1, self.n_vars))
            self.std = np.ones((1, self.n_vars))

        # Create sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create input-output sequence pairs."""
        sequences = []

        for i in range(len(self.normalized_data) - self.lookback - self.horizon + 1):
            x = self.normalized_data[i:i + self.lookback]  # (lookback, n_vars)
            y = self.normalized_data[i + self.lookback:i + self.lookback + self.horizon]  # (horizon, n_vars)
            sequences.append((x, y))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.sequences[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for learning relationships between variables."""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, dropout: float = 0.1):
        """
        Initialize graph attention layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.n_heads = n_heads
        self.out_features = out_features
        self.head_dim = out_features // n_heads

        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"

        # Linear transformations for Q, K, V
        self.q_linear = nn.Linear(in_features, out_features)
        self.k_linear = nn.Linear(in_features, out_features)
        self.v_linear = nn.Linear(in_features, out_features)

        # Output projection
        self.out_linear = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj_matrix: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-head attention.

        Args:
            x: Input tensor (batch_size, n_vars, features)
            adj_matrix: Optional adjacency matrix (n_vars, n_vars)

        Returns:
            Output tensor and attention weights
        """
        batch_size, n_vars, _ = x.shape

        # Linear projections
        Q = self.q_linear(x)  # (batch, n_vars, out_features)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Split into heads
        Q = Q.view(batch_size, n_vars, self.n_heads, self.head_dim).transpose(1, 2)  # (batch, heads, n_vars, head_dim)
        K = K.view(batch_size, n_vars, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_vars, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (batch, heads, n_vars, n_vars)

        # Apply adjacency matrix mask if provided
        if adj_matrix is not None:
            mask = adj_matrix.unsqueeze(0).unsqueeze(0)  # (1, 1, n_vars, n_vars)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, n_vars, n_vars)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (batch, heads, n_vars, head_dim)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, n_vars, self.out_features)

        # Output projection and residual connection
        out = self.out_linear(out)
        out = self.layer_norm(out + x) if x.shape[-1] == self.out_features else self.layer_norm(out)

        # Average attention weights across heads
        attn_weights_avg = attn_weights.mean(dim=1)  # (batch, n_vars, n_vars)

        return out, attn_weights_avg


class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for capturing time dependencies."""

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)

        Returns:
            Output tensor
        """
        # Self-attention over time dimension
        attn_out, _ = self.multihead_attn(x, x, x)
        attn_out = self.dropout(attn_out)

        # Residual connection and layer norm
        out = self.layer_norm(x + attn_out)

        return out


class KumoGraphTransformer(nn.Module):
    """
    Kumo Graph Transformer model for multivariate time series forecasting.

    Architecture:
    1. Variable embedding layer
    2. Graph attention layers to learn variable relationships
    3. Temporal attention layers for time dependencies
    4. Forecasting head
    """

    def __init__(
        self,
        n_vars: int,
        lookback: int,
        horizon: int,
        hidden_dim: int = 64,
        n_graph_layers: int = 2,
        n_temporal_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        learn_graph: bool = True
    ):
        """
        Initialize Kumo Graph Transformer.

        Args:
            n_vars: Number of variables
            lookback: Historical sequence length
            horizon: Forecast horizon
            hidden_dim: Hidden dimension size
            n_graph_layers: Number of graph attention layers
            n_temporal_layers: Number of temporal attention layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            learn_graph: Whether to learn graph structure from data
        """
        super().__init__()

        self.n_vars = n_vars
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.learn_graph = learn_graph

        # Variable embedding
        self.var_embedding = nn.Linear(1, hidden_dim)

        # Positional encoding for time
        self.pos_encoding = nn.Parameter(torch.randn(1, lookback, hidden_dim))

        # Graph structure learning
        if learn_graph:
            self.graph_learner = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_vars)
            )
        else:
            self.register_parameter('graph_learner', None)

        # Graph attention layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_graph_layers)
        ])

        # Temporal attention layers
        self.temporal_layers = nn.ModuleList([
            TemporalAttentionLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_temporal_layers)
        ])

        # Forecasting head
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim * lookback, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )

        self.dropout = nn.Dropout(dropout)

    def learn_adjacency_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Learn adjacency matrix from data.

        Args:
            x: Input tensor (batch_size, lookback, n_vars)

        Returns:
            Adjacency matrix (n_vars, n_vars)
        """
        if not self.learn_graph:
            # Return fully connected graph
            return torch.ones(self.n_vars, self.n_vars).to(x.device)

        batch_size = x.shape[0]

        # Embed variables
        x_mean = x.mean(dim=1)  # (batch, n_vars)
        x_embed = self.var_embedding(x_mean.unsqueeze(-1))  # (batch, n_vars, hidden_dim)

        # Compute pairwise relationships
        adj_logits = self.graph_learner(x_embed)  # (batch, n_vars, n_vars)

        # Average across batch and apply sigmoid
        adj_matrix = torch.sigmoid(adj_logits.mean(dim=0))  # (n_vars, n_vars)

        # Ensure symmetry
        adj_matrix = (adj_matrix + adj_matrix.T) / 2

        # Apply threshold for sparsity
        threshold = 0.5
        adj_matrix = (adj_matrix > threshold).float()

        # Add self-loops
        adj_matrix = adj_matrix + torch.eye(self.n_vars).to(x.device)

        return adj_matrix

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, lookback, n_vars)

        Returns:
            Forecast tensor (batch_size, horizon, n_vars) and metadata dict
        """
        batch_size = x.shape[0]

        # Learn or use fixed adjacency matrix
        adj_matrix = self.learn_adjacency_matrix(x)

        # Embed each variable's time series
        x_embed = self.var_embedding(x.unsqueeze(-1))  # (batch, lookback, n_vars, hidden_dim)

        # Add positional encoding
        x_embed = x_embed + self.pos_encoding.unsqueeze(2)  # (batch, lookback, n_vars, hidden_dim)

        # Process through graph attention layers (across variables)
        # Reshape to (batch * lookback, n_vars, hidden_dim)
        x_graph = x_embed.view(-1, self.n_vars, self.hidden_dim)

        graph_attn_weights = []
        for graph_layer in self.graph_layers:
            x_graph, attn_weights = graph_layer(x_graph, adj_matrix)
            graph_attn_weights.append(attn_weights)

        # Reshape back to (batch, lookback, n_vars, hidden_dim)
        x_graph = x_graph.view(batch_size, self.lookback, self.n_vars, self.hidden_dim)

        # Process each variable through temporal attention
        forecasts = []
        for var_idx in range(self.n_vars):
            x_var = x_graph[:, :, var_idx, :]  # (batch, lookback, hidden_dim)

            # Temporal attention
            for temporal_layer in self.temporal_layers:
                x_var = temporal_layer(x_var)

            # Flatten for forecasting head
            x_var_flat = x_var.view(batch_size, -1)  # (batch, lookback * hidden_dim)

            # Generate forecast
            forecast = self.forecast_head(x_var_flat)  # (batch, horizon)
            forecasts.append(forecast)

        # Stack forecasts
        forecasts = torch.stack(forecasts, dim=2)  # (batch, horizon, n_vars)

        # Metadata
        metadata = {
            'adjacency_matrix': adj_matrix,
            'graph_attention_weights': graph_attn_weights
        }

        return forecasts, metadata


class KumoForecaster(BaseForecaster):
    """
    Kumo Graph Transformer forecaster for multivariate economic time series.

    Automatically learns relationships between economic indicators and uses
    graph attention mechanisms for forecasting.
    """

    def __init__(
        self,
        name: str = 'KumoGraphTransformer',
        lookback: int = 12,
        horizon: int = 6,
        hidden_dim: int = 64,
        n_graph_layers: int = 2,
        n_temporal_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        n_epochs: int = 100,
        early_stopping_patience: int = 10,
        device: str = 'auto'
    ):
        """
        Initialize Kumo forecaster.

        Args:
            name: Forecaster name
            lookback: Number of historical steps to use
            horizon: Forecast horizon
            hidden_dim: Hidden dimension size
            n_graph_layers: Number of graph attention layers
            n_temporal_layers: Number of temporal attention layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        super().__init__(name)

        self.lookback = lookback
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.n_graph_layers = n_graph_layers
        self.n_temporal_layers = n_temporal_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.var_names = None
        self.learned_adjacency = None

    def fit(self, data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None, **kwargs) -> 'KumoForecaster':
        """
        Fit the Kumo Graph Transformer model.

        Args:
            data: DataFrame with time series for multiple variables (time x variables)
            val_data: Optional validation data
            **kwargs: Additional arguments

        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting {self.name} on {len(data)} samples with {data.shape[1]} variables")

        # Store variable names
        self.var_names = list(data.columns)
        n_vars = len(self.var_names)

        # Create dataset
        dataset = TimeSeriesGraphDataset(
            data=data,
            lookback=self.lookback,
            horizon=self.horizon,
            normalize=True
        )

        # Store scalers
        self.scaler_mean = dataset.mean
        self.scaler_std = dataset.std

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        # Initialize model
        self.model = KumoGraphTransformer(
            n_vars=n_vars,
            lookback=self.lookback,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            n_graph_layers=self.n_graph_layers,
            n_temporal_layers=self.n_temporal_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            learn_graph=True
        ).to(self.device)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        best_loss = float('inf')
        patience_counter = 0

        logger.info(f"Training on device: {self.device}")

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                predictions, metadata = self.model(batch_x)

                # Compute loss
                loss = criterion(predictions, batch_y)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            epoch_loss /= n_batches

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                # Save learned adjacency matrix
                self.model.eval()
                with torch.no_grad():
                    sample_x = next(iter(dataloader))[0].to(self.device)
                    self.learned_adjacency = self.model.learn_adjacency_matrix(sample_x).cpu().numpy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self.is_fitted = True
        self.training_data = data
        self.training_metadata = {
            'n_vars': n_vars,
            'var_names': self.var_names,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'n_epochs_trained': epoch + 1,
            'final_loss': epoch_loss,
            'best_loss': best_loss,
            'learned_adjacency': self.learned_adjacency.tolist() if self.learned_adjacency is not None else None
        }

        logger.info(f"{self.name} fitted successfully. Final loss: {epoch_loss:.6f}")

        return self

    def forecast(
        self,
        horizon: Optional[int] = None,
        confidence_level: float = 0.95,
        return_all_vars: bool = True
    ) -> Dict[str, ForecastResult]:
        """
        Generate forecasts for all variables.

        Args:
            horizon: Forecast horizon (uses training horizon if None)
            confidence_level: Confidence level for intervals
            return_all_vars: Whether to return forecasts for all variables

        Returns:
            Dictionary mapping variable names to ForecastResult objects
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")

        if horizon is None:
            horizon = self.horizon
        elif horizon != self.horizon:
            logger.warning(f"Requested horizon {horizon} differs from training horizon {self.horizon}. Using training horizon.")
            horizon = self.horizon

        self.model.eval()

        # Get last lookback window from training data
        last_window = self.training_data.iloc[-self.lookback:].values

        # Normalize
        last_window_norm = (last_window - self.scaler_mean) / self.scaler_std

        # Convert to tensor
        x = torch.FloatTensor(last_window_norm).unsqueeze(0).to(self.device)  # (1, lookback, n_vars)

        # Generate forecast
        with torch.no_grad():
            predictions, metadata = self.model(x)

        # Denormalize
        predictions_np = predictions.cpu().numpy()[0]  # (horizon, n_vars)
        predictions_denorm = predictions_np * self.scaler_std + self.scaler_mean

        # Create forecast results for each variable
        results = {}

        for var_idx, var_name in enumerate(self.var_names):
            point_forecast = predictions_denorm[:, var_idx]

            # Simple prediction intervals based on historical volatility
            historical_std = self.training_data[var_name].std()
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)

            # Expanding intervals
            margin = np.array([historical_std * z_score * np.sqrt(i + 1) for i in range(horizon)])
            lower_bound = point_forecast - margin
            upper_bound = point_forecast + margin

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
                    'adjacency_matrix': metadata['adjacency_matrix'].cpu().numpy().tolist(),
                    'variable_index': var_idx
                }
            )

        logger.info(f"Generated {horizon}-step forecast for {len(results)} variables")

        return results

    def get_learned_graph(self) -> pd.DataFrame:
        """
        Get the learned adjacency matrix as a DataFrame.

        Returns:
            DataFrame with learned variable relationships
        """
        if self.learned_adjacency is None:
            raise ValueError("Model must be fitted first")

        return pd.DataFrame(
            self.learned_adjacency,
            index=self.var_names,
            columns=self.var_names
        )

    def get_attention_weights(self, data: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Get attention weights for visualization.

        Args:
            data: Optional data to compute attention for (uses training data if None)

        Returns:
            Dictionary with attention weights
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if data is None:
            data = self.training_data.iloc[-self.lookback:]

        # Normalize
        data_norm = (data.values - self.scaler_mean) / self.scaler_std
        x = torch.FloatTensor(data_norm).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            _, metadata = self.model(x)

        # Extract attention weights
        graph_attn = [w.cpu().numpy() for w in metadata['graph_attention_weights']]

        return {
            'graph_attention': graph_attn,
            'adjacency_matrix': metadata['adjacency_matrix'].cpu().numpy()
        }
