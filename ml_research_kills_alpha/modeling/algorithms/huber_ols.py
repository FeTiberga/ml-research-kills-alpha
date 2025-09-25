import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.support.constants import RANDOM_SEED
from ml_research_kills_alpha.support import Logger


CROSS_VALIDATION_FOLDS = 2
MAX_ITER = 10000
SUB_SAMPLE_PCT = 0.2


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
        self.logger = Logger()
        
        self.is_deep = False  # flag to identify deep learning models for ensembling
        self.fixed_parameters = True # flag to identify whether we fine-tune hyperparameters once or for each time window
        
    def get_subsample(self, X_train, y_train, X_val, y_val, random_state=None):
        return super().get_subsample(SUB_SAMPLE_PCT, X_train, y_train, X_val, y_val, random_state)

    def train(self,
              X_train: pd.Series, y_train: pd.Series,
              X_val: pd.Series, y_val: pd.Series):

        if self.alpha is None or self.epsilon is None:
            self.logger.info(f"Hyperparameters not found for {self.name}")
            self.logger.info(f"Starting hyperparameter optimization")
            
            # get the subsample for hyperparameter tuning
            x_sub, y_sub = self.get_subsample(X_train, y_train, X_val, y_val)

            # grid search 0.0001 to 0.1 for alpha, 1.1 to 1.5 for epsilon
            param_grid = {
            'alpha': np.logspace(-4, -1, 5),
            'epsilon': np.linspace(1.1, 1.5, 5)
            }

            grid = GridSearchCV(estimator=HuberRegressor(max_iter=MAX_ITER),
                                param_grid=param_grid, cv=CROSS_VALIDATION_FOLDS)
            grid.fit(x_sub, y_sub)
            self.alpha = grid.best_params_['alpha']
            self.epsilon = grid.best_params_['epsilon']
            self.logger.info(f"Optimized {self.name} with alpha={self.alpha}, epsilon={self.epsilon}")

        # train the final model on all training data
        self.model = HuberRegressor(alpha=self.alpha,
                                    epsilon=self.epsilon,
                                    max_iter=MAX_ITER)
        # fit on train and validation data
        X_train = pd.concat([X_train, X_val], axis=0)
        y_train = pd.concat([y_train, y_val], axis=0)
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.Series) -> np.ndarray:
        a = self.model.predict(X)
        self.logger.info(f"Generated predictions for {self.name}")
        return a
    
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
