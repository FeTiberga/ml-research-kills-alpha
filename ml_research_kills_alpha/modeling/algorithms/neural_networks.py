import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.support.constants import RANDOM_SEED
from ml_research_kills_alpha.support import Logger

NEGATIVE_SLOPE = 0.3
TOL = 1e-6

# regularization
L1_NORMALIZATION_WEIGHT = 1e-5

# optimization
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.2
MIN_LEARNING_RATE = 1e-5
LR_PATIENCE = 10
BATCH_SIZE = 10000
MAXIMUM_EPOCHS = 200
EARLY_STOPPING = 20

# ensemble and configuration
NUM_ENSEMBLE = 5
CROSS_VALIDATION_FOLDS = 2
HYPERPARAMETER_SEARCH = True
torch.manual_seed(RANDOM_SEED)


class FFNNModel(Modeler):
    """
    Feed-Forward Neural Network model with N hidden layers.
    Layer sizes follow a geometric pyramid rule.
    Uses LeakyReLU activations and batch normalization in each hidden layer.
    Weight initialization with random normal. (using random seed from support.constants)
    Trained with early stopping, L1 regularization, and dynamic learning rate reduction.
    Each FFNN model averages an ensemble of 5 networks to reduce initialization variance.
    """
    def __init__(self,
                 num_layers: int,
                 input_dim: int | None = None, 
                 hidden_sizes: list[int] | None = None):
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        
        self.seed = RANDOM_SEED
        self.name = f"FFNN{num_layers}"
        self.logger = Logger()
        
        self.is_deep = True  # flag to identify deep learning models for ensembling
        self.fixed_parameters = False # flag to identify whether we fine-tune hyperparameters once or for each time window

        # build ensemble of {NUM_ENSEMBLE} networks
        self.nets: list[nn.Sequential] = []
        if input_dim is not None:
            for _ in range(NUM_ENSEMBLE):
                net = self._build_net()
                self.nets.append(net)
                
    def get_subsample(self):
        pass

    def _compute_hidden_sizes(self):
        # Compute geometric pyramid rule hidden layer sizes
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
            layers.append(nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE))
            in_dim = hid_dim

        # Output layer
        layers.append(nn.Linear(in_dim, 1))

        net = nn.Sequential(*layers)
        # Initialize weights
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        return net
    
    def _rebuild_if_needed(self, curr_in: int):
        """
        Rebuild ensemble if input_dim changed or nets are missing/mismatched.
        Also reset hidden_sizes so the geometric rule recomputes with the new input_dim.
        """
        need_rebuild = False

        if self.input_dim is None or self.input_dim != curr_in:
            self.input_dim = curr_in
            need_rebuild = True

        if self.nets and not need_rebuild:
            # check first Linear layer in the first net
            first_linear = next(m for m in self.nets[0] if isinstance(m, nn.Linear))
            if first_linear.in_features != self.input_dim:
                need_rebuild = True

        if need_rebuild:
            self.hidden_sizes = None  # force recompute with new input_dim
            self.nets = [self._build_net() for _ in range(NUM_ENSEMBLE)]

    def train(self, 
              X_train: pd.Series, y_train: pd.Series,
              X_val: pd.Series, y_val: pd.Series):

        # convert input data to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        torch.set_num_threads(int(os.getenv("PYTORCH_NUM_THREADS", "32")))
        
        # ensure architecture matches the CURRENT year’s feature count
        self._rebuild_if_needed(X_train.shape[1])

        # Build networks if not built yet
        if not self.nets:
            for _ in range(NUM_ENSEMBLE):
                net = self._build_net()
                self.nets.append(net)
            self.logger.info(f"Built {NUM_ENSEMBLE} networks with architecture: {net}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Train each network in the ensemble
        for i_net, net in enumerate(self.nets):
            net.to(device)
            net.train()
            optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=LEARNING_RATE_DECAY,
                                                                   patience=LR_PATIENCE,
                                                                   min_lr=MIN_LEARNING_RATE)
            criterion = nn.MSELoss()
            best_val_loss = float('inf')
            best_state = None
            epochs_no_improve = 0
            for _ in range(MAXIMUM_EPOCHS):

                # Shuffle training data each epoch
                permutation = np.random.permutation(len(X_train))
                batch_size = min(BATCH_SIZE, len(X_train))

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
                    loss = loss + L1_NORMALIZATION_WEIGHT * l1_penalty
                    loss.backward()
                    optimizer.step()

                # Compute validation loss for early stopping and LR scheduling
                net.eval()
                Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
                yv = torch.tensor(y_val, dtype=torch.float32, device=device)
                with torch.no_grad():
                    val_preds = net(Xv).squeeze(-1)
                    val_loss = criterion(val_preds, yv).item()
                net.train()

                scheduler.step(val_loss)
                if val_loss < best_val_loss - TOL:
                    best_val_loss = val_loss
                    best_state = {k: v.clone().cpu() for k, v in net.state_dict().items()}
                    epochs_no_improve = 0

                # Early stopping after {EARLY_STOPPING} epochs without improvement
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOPPING:
                    break
                
            # Load best model weights if early stopped
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            net.to(torch.device("cpu"))
            self.logger.info(f"Training complete for network {i_net+1}. Best validation loss: {best_val_loss:.6f}")

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
    
    def evaluate(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

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
