# ml_research_kills_alpha/modeling/elastic_net.py
# Elastic Net regression model for return prediction

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.support import Logger


CROSS_VALIDATION_FOLDS = 2
SUB_SAMPLE_PCT = 0.2


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
        self.logger = Logger()

        self.is_deep = False  # flag to identify deep learning models for ensembling
        self.fixed_parameters = True # flag to identify whether we fine-tune hyperparameters once or for each time window

    def get_subsample(self, X_train, y_train, X_val, y_val, random_state=None):
        return super().get_subsample(SUB_SAMPLE_PCT, X_train, y_train, X_val, y_val, random_state)

    def train(self,
              X_train: pd.Series, y_train: pd.Series,
              X_val: pd.Series, y_val: pd.Series):

        if self.alpha is None or self.l1_ratio is None:
            self.logger.info(f"Hyperparameters not found for {self.name}")
            self.logger.info(f"Starting hyperparameter optimization")
            
            # get the subsample for hyperparameter tuning
            x_sub, y_sub = self.get_subsample(X_train, y_train, X_val, y_val)

            # grid search 0.0001 to 1 for alpha, 0.1 to 0.9 for l1_ratio
            param_grid = {
                'alpha': np.logspace(-4, 0, 5),
                'l1_ratio': np.linspace(0.1, 0.9, 9)
            }

            grid = GridSearchCV(ElasticNet(), param_grid, cv=CROSS_VALIDATION_FOLDS)
            grid.fit(x_sub, y_sub)
            self.alpha = grid.best_params_['alpha']
            self.l1_ratio = grid.best_params_['l1_ratio']
            self.logger.info(f"Optimized {self.name} with alpha={self.alpha}, l1_ratio={self.l1_ratio}")

        # Train final ElasticNet model with determined hyperparameters
        self.model = ElasticNet(alpha=self.alpha,
                                l1_ratio=self.l1_ratio)
        # fit on train and validation data
        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        self.logger.info(f"Generated predictions for {self.name}")
        return preds
    
    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(np.abs(preds - y))

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
