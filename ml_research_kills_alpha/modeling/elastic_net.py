# ml_research_kills_alpha/modeling/elastic_net.py
# Elastic Net regression model for return prediction

import numpy as np
import joblib

from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import GridSearchCV

from ml_research_kills_alpha.modeling.base_model import Modeler


class ElasticNetModel(Modeler):

    def __init__(self, alpha=None, l1_ratio=None, cv_folds=2):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.model = None
        self.name = "ENET"
        self.is_deep = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on given training data, optionally using validation data for tuning.
        
        Args:
            X_train (pd.Series): Features for training.
            y_train (pd.Series): Target variable for training.
            X_val (pd.Series): Optional features for validation.
            y_val (pd.Series): Optional target variable for validation.
        """
        if self.alpha is None or self.l1_ratio is None:
            if X_val is not None and y_val is not None:
                param_grid = {
                    'alpha': np.logspace(-4, 0, 10),
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
                grid = GridSearchCV(
                    ElasticNet(max_iter=10000),
                    param_grid,
                    cv=[(np.arange(len(X_train)), np.arange(len(X_train), len(X_train) + len(X_val)))],
                    n_jobs=-1
                )
                X_combined = np.vstack([X_train, X_val])
                y_combined = np.concatenate([y_train, y_val])
                grid.fit(X_combined, y_combined)
                self.alpha = grid.best_params_['alpha']
                self.l1_ratio = grid.best_params_['l1_ratio']
            else:
                enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-4, 0, 10),
                                       cv=self.cv_folds, n_jobs=-1, max_iter=10000)
                enet_cv.fit(X_train, y_train)
                self.alpha = enet_cv.alpha_
                self.l1_ratio = enet_cv.l1_ratio_
        # Train final ElasticNet model with determined hyperparameters
        self.model = ElasticNet(alpha=self.alpha if self.alpha is not None else 1.0,
                                l1_ratio=self.l1_ratio if self.l1_ratio is not None else 0.5,
                                max_iter=10000)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
