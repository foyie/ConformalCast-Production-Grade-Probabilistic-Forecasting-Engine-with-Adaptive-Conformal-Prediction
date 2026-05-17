"""
LSTM with Monte Carlo Dropout
================================
MC Dropout (Gal & Ghahramani, 2016): keep dropout ACTIVE at inference
time and run T forward passes. The empirical distribution over T
predictions approximates the Bayesian posterior predictive.

This gives us:
  - mean(T predictions) → point forecast
  - std(T predictions) → aleatoric + epistemic uncertainty
  - percentiles(T predictions) → prediction intervals

Why not just use softmax confidence? Because softmax confidence is
notoriously overconfident. MC Dropout is calibrated.
"""

import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import DataLoader, TensorDataset


class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM with dropout on all layers.

    Architecture: LSTM(input→hidden) → Dropout → LSTM(hidden→hidden) → Dropout → Linear(hidden→1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Explicit dropout layer for MC Dropout at inference
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Take last time step
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)

        return self.fc(last_out).squeeze(-1)

    def enable_dropout(self):
        """Force dropout layers active (for MC Dropout inference)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class MCDropoutForecaster:
    """
    Wraps LSTMForecaster with Monte Carlo Dropout inference.

    Usage:
        forecaster = MCDropoutForecaster(...)
        forecaster.fit(X_train, y_train)
        mean, std, lower, upper = forecaster.predict_with_uncertainty(X_test)
    """

    def __init__(
        self,
        input_size: int,
        sequence_length: int = 168,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        mc_samples: int = 100,
        device: Optional[str] = None,
    ):
        self.sequence_length = sequence_length
        self.mc_samples = mc_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.criterion = nn.MSELoss()

        self.scaler_mean: Optional[float] = None
        self.scaler_std: Optional[float] = None

    def _make_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Convert flat features to (batch, seq_len, features) sequences."""
        n = len(X)
        sequences = []
        targets = []

        for i in range(self.sequence_length, n):
            sequences.append(X[i - self.sequence_length : i])
            if y is not None:
                targets.append(y[i])

        X_seq = np.array(sequences, dtype=np.float32)

        if y is not None:
            y_seq = np.array(targets, dtype=np.float32)
            return X_seq, y_seq
        return X_seq

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MCDropoutForecaster":
        """Train LSTM. Normalizes target internally."""

        # Normalize target
        self.scaler_mean = y_train.mean()
        self.scaler_std = y_train.std()
        y_train_norm = (y_train - self.scaler_mean) / self.scaler_std

        X_seq, y_seq = self._make_sequences(X_train, y_train_norm)

        dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.FloatTensor(y_seq),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            y_val_norm = (y_val - self.scaler_mean) / self.scaler_std
            X_val_seq, y_val_seq = self._make_sequences(X_val, y_val_norm)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq),
                torch.FloatTensor(y_val_seq),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 4)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(loader)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        pred = self.model(X_batch.to(self.device))
                        val_loss += self.criterion(pred, y_batch.to(self.device)).item()
                avg_val_loss = val_loss / len(val_loader)
                self.scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), "/tmp/lstm_best.pt")
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"    Epoch {epoch+1}/{self.epochs} | train={avg_train_loss:.4f} | val={avg_val_loss:.4f}")

                if patience_counter >= 10:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        # Load best checkpoint
        if val_loader is not None:
            self.model.load_state_dict(torch.load("/tmp/lstm_best.pt", map_location=self.device))

        return self

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        coverage: float = 0.80,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        MC Dropout inference.

        Returns:
            mean: point forecast
            std: predictive standard deviation
            lower: lower quantile at given coverage
            upper: upper quantile at given coverage
        """
        X_seq = self._make_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        # Set eval mode but keep dropout ACTIVE
        self.model.eval()
        self.model.enable_dropout()

        sample_predictions = []

        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.model(X_tensor).cpu().numpy()
                # Denormalize
                pred = pred * self.scaler_std + self.scaler_mean
                sample_predictions.append(pred)

        samples = np.stack(sample_predictions, axis=0)  # (mc_samples, n)

        alpha = (1 - coverage) / 2
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        lower = np.percentile(samples, alpha * 100, axis=0)
        upper = np.percentile(samples, (1 - alpha) * 100, axis=0)

        return mean, std, lower, upper

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/lstm_weights.pt")
        joblib.dump({
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "sequence_length": self.sequence_length,
            "mc_samples": self.mc_samples,
        }, f"{path}/lstm_meta.pkl")
        print(f"LSTM saved to {path}/")

    @classmethod
    def load(cls, path: str, input_size: int) -> "MCDropoutForecaster":
        meta = joblib.load(f"{path}/lstm_meta.pkl")
        # Filter out input_size from meta if it exists (avoid duplicate kwargs)
        meta_kwargs = {k: v for k, v in meta.items() if k != "input_size"}
        forecaster = cls(input_size=input_size, **meta_kwargs)
        forecaster.model.load_state_dict(
            torch.load(f"{path}/lstm_weights.pt", map_location=forecaster.device)
        )
        return forecaster
# # Replace with your actual Render URL
# RENDER_URL="https://conformalcast-probabilistic-forecasting.onrender.com"

# curl $RENDER_URL/

# curl $RENDER_URL/health

# curl $RENDER_URL/models

# curl -X POST $RENDER_URL/forecast \
#   -H "Content-Type: application/json" \
#   -d '{"horizon": 24, "coverage": 0.80}'
# curl $RENDER_URL/monitoring/health
