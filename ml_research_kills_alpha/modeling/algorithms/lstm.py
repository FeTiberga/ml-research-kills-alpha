# Long Short-Term Memory (LSTM) model for return prediction

from __future__ import annotations
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.support.constants import RANDOM_SEED, LSTM_SEQUENCE_LENGTH
from ml_research_kills_alpha.support import Logger

# Training knobs
LEARNING_RATE = 0.01
LR_DECAY = 0.2
LR_PATIENCE = 10
MIN_LR = 1e-5
MAX_EPOCHS = 200
EARLY_STOPPING = 20
L1_WEIGHT = 1e-5
NUM_ENSEMBLE = 5


class LSTMModel(Modeler):
    """
    LSTM regressor that consumes fixed-length monthly sequences per stock, and
    averages an ensemble of identical nets to reduce init variance.

    Args:
        num_layers: 1 or 2 LSTM layers.
        hidden_sizes: List of hidden sizes; if None, computed by a geometric rule.
        input_dim: Feature dimension per month (D).
        seq_length: Number of months in the input sequence (T).
    """
    def __init__(self,
                 num_layers: int = 1,
                 hidden_sizes: List[int] | None = None,
                 input_dim: int | None = None):
        super().__init__(name=f"LSTM{num_layers}")
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.seq_length = LSTM_SEQUENCE_LENGTH
        self.nets: list[LSTMNet] = []
        self.logger = Logger()
        self.is_deep = True
        self.fixed_parameters = False
        torch.manual_seed(RANDOM_SEED)
        if self.input_dim is not None and self.hidden_sizes is not None:
            self.nets = [self._build_net() for _ in range(NUM_ENSEMBLE)]

    def get_subsample(self):
        """Not used for LSTM; kept for interface symmetry."""
        pass

    def _compute_hidden_sizes(self) -> List[int]:
        """Geometric pyramid rule used elsewhere in the repo for deep nets."""
        L, I = self.num_layers, self.input_dim
        sizes = []
        for j in range(1, L + 1):
            size = int(round(I ** (1 - j / (L + 1))))
            sizes.append(max(size, 1))
        return sizes

    def _build_net(self) -> "LSTMNet":
        if self.hidden_sizes is None:
            self.hidden_sizes = self._compute_hidden_sizes()
        return LSTMNet(self.input_dim, self.hidden_sizes, self.num_layers, dropout_prob=0.2)

    def _ensure_nets(self, current_input_dim: int) -> None:
        """Rebuild ensemble if input_dim changed (rolling windows may add/drop features)."""
        if self.input_dim != current_input_dim or not self.nets:
            self.input_dim = current_input_dim
            self.hidden_sizes = None  # force recompute
            self.nets = [self._build_net() for _ in range(NUM_ENSEMBLE)]

    def train(self,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Fit the LSTM ensemble.

        Args:
            X_train: shape (N_train, seq_length, input_dim)
            y_train: shape (N_train,)
            X_val:   shape (N_val,   seq_length, input_dim)
            y_val:   shape (N_val,)
        """
        X_train = np.asarray(X_train); y_train = np.asarray(y_train)
        X_val = np.asarray(X_val);     y_val = np.asarray(y_val)

        assert X_train.ndim == 3, "LSTM expects 3D input: (N, T, D)"
        self._ensure_nets(X_train.shape[2])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i_net, net in enumerate(self.nets):
            net.to(device)
            net.train()

            optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=LR_DECAY, patience=LR_PATIENCE, min_lr=MIN_LR
            )
            criterion = nn.MSELoss()
            best_val = float("inf"); best_state = None; epochs_no_improve = 0

            for _ in range(MAX_EPOCHS):
                # Single-batch training is OK for large N; you can add mini-batching if needed.
                Xb = torch.tensor(X_train, dtype=torch.float32, device=device)
                yb = torch.tensor(y_train, dtype=torch.float32, device=device)

                optimizer.zero_grad()
                preds = net(Xb).squeeze(-1)
                loss = criterion(preds, yb)
                # L1 regularization on weights
                l1 = sum(torch.norm(p, 1) for p in net.parameters())
                (loss + L1_WEIGHT * l1).backward()
                optimizer.step()

                # Validation
                net.eval()
                Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
                yv = torch.tensor(y_val, dtype=torch.float32, device=device)
                with torch.no_grad():
                    vpred = net(Xv).squeeze(-1)
                    vloss = criterion(vpred, yv).item()
                net.train()

                scheduler.step(vloss)
                if vloss + 1e-6 < best_val:
                    best_val = vloss
                    best_state = {k: v.clone().cpu() for k, v in net.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING:
                    break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            net.to(torch.device("cpu"))
            self.logger.info(f"LSTM net {i_net+1}/{NUM_ENSEMBLE} best val MSE: {best_val:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                preds.append(net(X_tensor).squeeze(-1).numpy())
        return np.mean(preds, axis=0)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def save(self, filepath: str) -> None:
        data = {
            "input_dim": self.input_dim,
            "num_layers": self.num_layers,
            "hidden_sizes": self.hidden_sizes,
            "seq_length": self.seq_length,
            "state_dicts": [net.state_dict() for net in self.nets],
        }
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "LSTMModel":
        data = torch.load(filepath, map_location="cpu")
        model = cls(num_layers=data["num_layers"], hidden_sizes=data["hidden_sizes"],
                    input_dim=data["input_dim"], seq_length=data.get("seq_length", 12))
        for net, state in zip(model.nets, data["state_dicts"]):
            net.load_state_dict(state)
            net.eval()
        return model


class LSTMNet(nn.Module):
    """Two‑option LSTM body (1‑layer or 2‑layer) with BN and a final linear head."""
    def __init__(self, input_dim: int, hidden_sizes: List[int], num_layers: int, dropout_prob: float):
        super().__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_sizes[0], batch_first=True)
        else:
            self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_sizes[0], batch_first=True)
            self.dropout = nn.Dropout(dropout_prob)
            self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)

        self.bn_final = nn.BatchNorm1d(hidden_sizes[-1])
        self.fc = nn.Linear(hidden_sizes[-1], 1)

        # Random normal init like FFNN
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param)
            if "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_length, input_dim)
        if self.num_layers == 1:
            out, _ = self.lstm1(x)
        else:
            out1, _ = self.lstm1(x)
            out1 = self.dropout(out1)
            out, _ = self.lstm2(out1)
        last = out[:, -1, :]
        last = self.bn_final(last)
        return self.fc(last)
