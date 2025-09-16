# ml_research_kills_alpha/modeling/elastic_net.py
# Elastic Net regression model for return prediction

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler


class ElasticNetModel(Modeler):
    """
    Elastic Net regression model: linear model that combines L1 and L2 regularization.
    Useful when there are multiple features correlated with each other.
    Loss is defined as:
    (1 / (2n)) * ||y - Xr||²₂ + alpha * [ (1 - l1_ratio) * (||r||²₂ / 2) + l1_ratio * ||r||₁ ]

    Args:
        alpha (float): L2 regularization penalty.
                       If not given, it will be determined by GridSearchCV.
        l1_ratio (float): The mixing parameter between L1 and L2 regularization. 
                          If not given, it will be determined by GridSearchCV.
    """
    def __init__(self, alpha=None, l1_ratio=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = None
        self.name = "ENET"

    def train(self,
              X_train: pd.Series, y_train: pd.Series,
              X_val: pd.Series, y_val: pd.Series):

        if self.alpha is None or self.l1_ratio is None:
            param_grid = {
                'alpha': np.logspace(-4, 0, 5),
                'l1_ratio': np.linspace(0.1, 0.9, 9)
            }
            grid = GridSearchCV(ElasticNet(), param_grid, cv=2)
            grid.fit(X_val, y_val)
            self.alpha = grid.best_params_['alpha']
            self.l1_ratio = grid.best_params_['l1_ratio']

        # Train final ElasticNet model with determined hyperparameters
        self.model = ElasticNet(alpha=self.alpha,
                                l1_ratio=self.l1_ratio)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(np.abs(preds - y))

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
