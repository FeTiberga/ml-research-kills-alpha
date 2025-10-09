# Lasso regression model for return prediction

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.support import Logger


CROSS_VALIDATION_FOLDS = 2


class LassoModel(Modeler):
    """
    Lasso regression model: linear model that uses L1 regularization.
    Useful when we want to enforce sparsity in the model.
    Loss is defined as:
    (1 / (2n)) * ||y - Xr||²₂ + alpha * ||r||₁

    Args:
        alpha (float): L1 regularization penalty.
                       If not given, it will be determined by GridSearchCV.
    """
    def __init__(self, alpha=None):
        self.alpha = alpha
        self.model = None
        self.name = "LASSO"
        self.logger = Logger()

        self.is_deep = False  # flag to identify deep learning models for ensembling
        self.fixed_parameters = True # flag to identify whether we fine-tune hyperparameters once or for each time window

    def get_subsample(self, X_train, y_train, X_val, y_val):
        return super().get_subsample(X_train, y_train, X_val, y_val)

    def train(self,
              X_train: pd.Series, y_train: pd.Series,
              X_val: pd.Series, y_val: pd.Series):

        if self.alpha is None:
            self.logger.info(f"Hyperparameters not found for {self.name}")
            self.logger.info(f"Starting hyperparameter optimization")
            
            # get the subsample for hyperparameter tuning
            x_sub, y_sub = self.get_subsample(X_train, y_train, X_val, y_val)

            # grid search 0.0001 to 1 for alpha
            param_grid = {
                'alpha': np.logspace(-4, 0, 5)
            }

            grid = GridSearchCV(Lasso(), param_grid, cv=CROSS_VALIDATION_FOLDS)
            grid.fit(x_sub, y_sub)
            self.alpha = grid.best_params_['alpha']
            self.logger.info(f"Optimized {self.name} with alpha={self.alpha}")

        # Train final Lasso model with determined hyperparameters
        self.model = Lasso(alpha=self.alpha)
        # fit on train and validation data
        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        preds = self.model.predict(X)
        return preds

    def evaluate(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
