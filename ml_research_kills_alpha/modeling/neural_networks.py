import numpy as np
import torch
import torch.nn as nn
from ml_research_kills_alpha.modeling.base_model import Modeler


class FFNNModel(Modeler):
    """Feed-Forward Neural Network model with 2-5 hidden layers.
    Uses LeakyReLU activations and batch normalization in each hidden layer.
    Trained with early stopping, L1 regularization, and dynamic learning rate reduction.
    Each FFNN model averages an ensemble of 5 networks to reduce initialization variance.
    """
    def __init__(self, num_layers=3, hidden_sizes=None, input_dim=None):
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.nets: list[nn.Sequential] = []
        self.name = f"FFNN{num_layers}"
        self.is_deep = True
        # If input_dim and hidden_sizes are provided (e.g. when loading), build networks immediately
        if input_dim is not None and hidden_sizes is not None:
            for _ in range(5):
                net = self._build_net()
                self.nets.append(net)

    def _compute_hidden_sizes(self):
        # Compute geometric pyramid rule hidden layer sizes:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}
        L = self.num_layers
        I = self.input_dim
        sizes = []
        for j in range(1, L+1):
            size = int(round(I ** (1 - j/(L+1))))
            if size < 1:
                size = 1
            sizes.append(size)
        return sizes

    def _build_net(self) -> nn.Sequential:
        # Build a feed-forward network with the specified architecture
        if self.hidden_sizes is None:
            self.hidden_sizes = self._compute_hidden_sizes()
        layers = []
        in_dim = self.input_dim
        # Hidden layers
        for hid_dim in self.hidden_sizes:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(nn.BatchNorm1d(hid_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.3))
            in_dim = hid_dim
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        net = nn.Sequential(*layers)
        # Initialize weights (random normal):contentReference[oaicite:9]{index=9}
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        return net

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Ensure numpy arrays for training data
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val) if X_val is not None else None
        y_val = np.array(y_val) if y_val is not None else None
        # Set input_dim if not already set
        if self.input_dim is None:
            self.input_dim = X_train.shape[1]
        # Build networks if not built yet
        if not self.nets:
            for _ in range(5):
                net = self._build_net()
                self.nets.append(net)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Train each network in the ensemble
        for net in self.nets:
            net.to(device)
            net.train()
            optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=10, min_lr=1e-5)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            best_state = None
            epochs_no_improve = 0
            for epoch in range(200):
                # Shuffle training data each epoch
                permutation = np.random.permutation(len(X_train))
                batch_size = 10000
                for i in range(0, len(X_train), batch_size):
                    idx = permutation[i:i+batch_size]
                    batch_X = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
                    batch_y = torch.tensor(y_train[idx], dtype=torch.float32, device=device)
                    optimizer.zero_grad()
                    preds = net(batch_X).squeeze(-1)
                    loss = criterion(preds, batch_y)
                    # L1 regularization penalty
                    l1_penalty = 0.0
                    for param in net.parameters():
                        l1_penalty += torch.norm(param, 1)
                    loss = loss + 1e-5 * l1_penalty
                    loss.backward()
                    optimizer.step()
                # Compute validation loss for early stopping and LR scheduling
                if X_val is not None and y_val is not None:
                    net.eval()
                    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
                    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        val_preds = net(Xv).squeeze(-1)
                        val_loss = criterion(val_preds, yv).item()
                    net.train()
                else:
                    val_loss = 0.0
                scheduler.step(val_loss)
                if X_val is not None and y_val is not None:
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = {k: v.clone().cpu() for k, v in net.state_dict().items()}
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= 20:
                        # Early stopping after 20 epochs without improvement
                        break
            # Load best model weights if early stopped
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            net.to(torch.device("cpu"))

    def predict(self, X):
        X = np.array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Average predictions from all 5 networks
        preds_list = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                pred = net(X_tensor).squeeze(-1).numpy()
                preds_list.append(pred)
        preds_mean = np.mean(preds_list, axis=0)
        return preds_mean

    def save(self, filepath):
        # Save model configuration and state
        data = {
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'hidden_sizes': self.hidden_sizes,
            'state_dicts': [net.state_dict() for net in self.nets]
        }
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath):
        data = torch.load(filepath, map_location='cpu')
        model = cls(num_layers=data['num_layers'], hidden_sizes=data['hidden_sizes'], input_dim=data['input_dim'])
        # Load weights into each network
        for net, state_dict in zip(model.nets, data['state_dicts']):
            net.load_state_dict(state_dict)
            net.eval()
        return model
