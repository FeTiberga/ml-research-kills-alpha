import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler


class HuberRegressorModel(Modeler):
    """
    Huber regression model: linear model where the loss function is
    quadratic for small errors and linear for large errors, making it robust to outliers.
    
    Defined as:
        ℓₑ(r) = 0.5 * r²                  if |r| ≤ ε
                ε * (|r| - 0.5 * ε)       if |r| > ε
    Args:
        alpha (float): L2 regularization penalty.
                       If not given, it will be determined by GridSearchCV.
        epsilon (float): Huber threshold for outliers.
                         If not given, it will be determined by GridSearchCV.
    """
    def __init__(self,
                 alpha: float = None, epsilon: float = None):
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = None
        self.name = "OLS-H"

    def train(self,
              X_train: pd.Series, y_train: pd.Series,
              X_val: pd.Series, y_val: pd.Series):

        # If hyperparameters not specified, perform grid search 
        # to find best alpha and epsilon using validation set
        if self.alpha is None or self.epsilon is None:
            # grid search 0.0001 to 0.1 for alpha, 1.1 to 1.5 for epsilon
            param_grid = {
            'alpha': np.logspace(-4, -1, 5),
            'epsilon': np.linspace(1.1, 1.5, 5)
            }
            grid = GridSearchCV(estimator=HuberRegressor(), param_grid=param_grid, cv=2)
            grid.fit(X_val, y_val)
            self.alpha = grid.best_params_['alpha']
            self.epsilon = grid.best_params_['epsilon']
        
        # train the final model on all training data
        self.model = HuberRegressor(alpha=self.alpha,
                                    epsilon=self.epsilon)
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.Series) -> np.ndarray:
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        # return huber loss on given data
        preds = self.predict(X)
        residual = preds - y
        is_small_error = np.abs(residual) <= self.epsilon
        return np.mean(np.where(is_small_error, 0.5 * residual**2, self.epsilon * (np.abs(residual) - 0.5 * self.epsilon)))

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
