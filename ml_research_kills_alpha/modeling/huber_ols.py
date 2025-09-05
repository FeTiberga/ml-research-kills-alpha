import numpy as np
import joblib
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV

from ml_research_kills_alpha.modeling.base_model import Modeler


class HuberRegressorModel(Modeler):
    """Huber regression model (robust linear model) for return prediction:contentReference[oaicite:2]{index=2}."""
    def __init__(self, alpha=None, epsilon=None, cv_folds=2):
        self.alpha = alpha
        self.epsilon = epsilon
        self.cv_folds = cv_folds
        self.model = None
        self.name = "OLS-H"  # OLS-Huber as referred in the paper
        self.is_deep = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # If hyperparameters not specified, perform grid search (2-fold CV) to find best alpha and epsilon
        if self.alpha is None or self.epsilon is None:
            param_grid = {
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'epsilon': [1.1, 1.2, 1.35, 1.5]
            }
            grid = GridSearchCV(HuberRegressor(max_iter=10000), param_grid, cv=self.cv_folds,
                                scoring='neg_mean_squared_error', n_jobs=-1)
            grid.fit(X_train, y_train)
            self.alpha = grid.best_params_['alpha']
            self.epsilon = grid.best_params_['epsilon']
        # Train final HuberRegressor with best (or given) parameters
        self.model = HuberRegressor(alpha=self.alpha if self.alpha is not None else 0.0,
                                    epsilon=self.epsilon if self.epsilon is not None else 1.35,
                                    max_iter=10000)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
