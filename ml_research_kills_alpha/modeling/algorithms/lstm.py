import numpy as np
import torch
import torch.nn as nn
from .base_model import Modeler


class LSTMModel(Modeler):
    """Long Short-Term Memory (LSTM) model with 1-2 LSTM layers.
    Uses 12-month input sequences for each stock:contentReference[oaicite:12]{index=12}.
    Includes dropout (0.2) on LSTM layers and batch normalization on output layer.
    Trained with L1 regularization, early stopping, and learning rate scheduling similar to FFNN.
    Each LSTM model averages an ensemble of 5 networks for stability.
    """
    def __init__(self, num_layers=1, hidden_sizes=None, input_dim=None, seq_length=12):
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.nets: list['LSTMNet'] = []
        self.name = f"LSTM{num_layers}"
        self.is_deep = True  # flag to identify deep learning models for ensembling
        self.fixed_parameters = False # flag to identify whether we fine-tune hyperparameters once or for each time window
        
        if input_dim is not None and hidden_sizes is not None:
            for _ in range(5):
                net = self._build_net()
                self.nets.append(net)
                
    def get_subsample(self):
        pass

    def _compute_hidden_sizes(self):
        # Geometric pyramid rule for LSTM hidden layer sizes
        L = self.num_layers
        I = self.input_dim
        sizes = []
        for j in range(1, L+1):
            size = int(round(I ** (1 - j/(L+1))))
            if size < 1:
                size = 1
            sizes.append(size)
        return sizes

    def _build_net(self) -> 'LSTMNet':
        # Define an LSTM-based neural network as an inner class
        if self.hidden_sizes is None:
            self.hidden_sizes = self._compute_hidden_sizes()

        # Instantiate the LSTMNet
        net = LSTMNet(self.input_dim, self.hidden_sizes, self.num_layers, dropout_prob=0.2)
        return net

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # X_train should be an array of shape (N, seq_length, input_dim)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val) if X_val is not None else None
        y_val = np.array(y_val) if y_val is not None else None
        if self.input_dim is None:
            self.input_dim = X_train.shape[2]
        if self.hidden_sizes is None:
            self.hidden_sizes = self._compute_hidden_sizes()
        # Build networks if not already built
        if not self.nets:
            for _ in range(5):
                net = self._build_net()
                self.nets.append(net)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                permutation = np.random.permutation(len(X_train))
                batch_size = 10000
                for i in range(0, len(X_train), batch_size):
                    idx = permutation[i:i+batch_size]
                    batch_X = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
                    batch_y = torch.tensor(y_train[idx], dtype=torch.float32, device=device)
                    optimizer.zero_grad()
                    preds = net(batch_X).squeeze(-1)
                    loss = criterion(preds, batch_y)
                    # L1 regularization
                    l1_penalty = 0.0
                    for param in net.parameters():
                        l1_penalty += torch.norm(param, 1)
                    loss = loss + 1e-5 * l1_penalty
                    loss.backward()
                    optimizer.step()
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
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            net.to(torch.device("cpu"))

    def predict(self, X):
        X = np.array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        preds_list = []
        for net in self.nets:
            net.eval()
            with torch.no_grad():
                pred = net(X_tensor).squeeze(-1).numpy()
                preds_list.append(pred)
        preds_mean = np.mean(preds_list, axis=0)
        return preds_mean

    def save(self, filepath):
        data = {
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'hidden_sizes': self.hidden_sizes,
            'seq_length': self.seq_length,
            'state_dicts': [net.state_dict() for net in self.nets]
        }
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath):
        data = torch.load(filepath, map_location='cpu')
        model = cls(num_layers=data['num_layers'], hidden_sizes=data['hidden_sizes'],
                    input_dim=data['input_dim'], seq_length=data.get('seq_length', 12))
        for net, state_dict in zip(model.nets, data['state_dicts']):
            net.load_state_dict(state_dict)
            net.eval()
        return model


class LSTMNet(nn.Module):
    def __init__(inner_self, input_dim, hidden_sizes, num_layers, dropout_prob):
        super().__init__()
        inner_self.num_layers = num_layers
        # Define LSTM layers
        if num_layers == 1:
            inner_self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_sizes[0], batch_first=True)
        else:
            inner_self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_sizes[0], batch_first=True)
            inner_self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
            inner_self.dropout = nn.Dropout(dropout_prob)
        inner_self.bn_final = nn.BatchNorm1d(hidden_sizes[-1])
        inner_self.fc = nn.Linear(hidden_sizes[-1], 1)
        # Initialize weights (random normal)
        for name, param in inner_self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
    def forward(inner_self, x):
        # x shape: (batch, seq_length, input_dim)
        if inner_self.num_layers == 1:
            out, _ = inner_self.lstm1(x)
        else:
            out1, _ = inner_self.lstm1(x)
            out1_d = inner_self.dropout(out1)
            out2, _ = inner_self.lstm2(out1_d)
            out = out2
        # Take output of the last time step
        out_last = out[:, -1, :]
        # Batch normalization on final hidden representation
        out_last = inner_self.bn_final(out_last)
        # Final linear output
        return inner_self.fc(out_last)
