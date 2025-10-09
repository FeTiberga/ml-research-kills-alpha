# Random Forest model for return prediction

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.support.constants import RANDOM_SEED
from ml_research_kills_alpha.support import Logger


CROSS_VALIDATION_FOLDS = 2
N_ITER = 25


class RandomForestModel(Modeler):
    def __init__(self):
        super().__init__(name="RF")
        self.logger = Logger()
        self.seed = RANDOM_SEED
        self.is_deep = False           # not deep
        self.fixed_parameters = False  # tune each window
        self.params: dict[str, float|int] = {}  # filled by _tune

    def _predefined_split(self, X_train, X_val):
        test_fold = np.r_[np.full(len(X_train), -1, dtype=int),
                          np.zeros(len(X_val), dtype=int)]
        return PredefinedSplit(test_fold=test_fold)

    def _tune(self, X_train, y_train, X_val, y_val) -> None:
        X_all = pd.concat([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0)
        ps = self._predefined_split(X_train, X_val)

        base = RandomForestRegressor(random_state=self.seed, oob_score=False)
        space = {
            "n_estimators": [200, 400, 800, 1200],
            "max_depth": [None, 6, 12, 18, 24],
            "max_features": ["sqrt", 0.3, 0.5, 0.8],
            "min_samples_leaf": [1, 2, 4, 8]
        }
        search = RandomizedSearchCV(
            estimator=base, param_distributions=space,
            n_iter=N_ITER, cv=ps, scoring="neg_mean_squared_error",
            random_state=self.seed, refit=False)
        search.fit(X_all, y_all)
        self.params = search.best_params_
        self.logger.info(f"[RF] tuned params -> {self.params}")

    def train(self, X_train, y_train, X_val, y_val) -> None:
        self._tune(X_train, y_train, X_val, y_val)
        X_all = pd.concat([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0)
        self.model = RandomForestRegressor(random_state=self.seed, **self.params)
        self.model.fit(X_all, y_all)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def save(self, filepath: str) -> None:
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "RandomForestModel":
        return joblib.load(filepath)
