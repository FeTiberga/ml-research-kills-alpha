# ml_research_kills_alpha/modeling/algorithms/xgboost.py

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from xgboost import XGBRegressor

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.support.constants import RANDOM_SEED
from ml_research_kills_alpha.support import Logger

N_ITER = 25
EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS = 2000

class XGBoostModel(Modeler):
    """
    XGBoost regressor tuned every window using the validation set.

    We search on a single train/val split via PredefinedSplit, then
    refit with early stopping to choose the best number of trees.
    """
    def __init__(self):
        super().__init__(name="XGB")
        self.logger = Logger()
        self.seed = RANDOM_SEED
        self.is_deep = False           # tree models are NOT 'deep'
        self.fixed_parameters = False  # tune each window
        self.params: dict[str, float|int] = {}

    def _predefined_split(self, X_train, X_val):
        """Create a PredefinedSplit that trains on train rows and scores on val rows."""
        n_tr, n_val = len(X_train), len(X_val)
        test_fold = np.r_[np.full(n_tr, -1, dtype=int), np.zeros(n_val, dtype=int)]
        return PredefinedSplit(test_fold=test_fold)

    def _tune(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame,   y_val: pd.Series) -> None:
        """
        Randomized search over structural params using PredefinedSplit.
        """
        X_all = pd.concat([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0)
        ps = self._predefined_split(X_train, X_val)

        base = XGBRegressor(
            n_estimators=N_ESTIMATORS,
            objective="reg:squarederror",
            random_state=self.seed,
            tree_method="hist")
        space = {
            "learning_rate": [0.02, 0.04, 0.06, 0.08, 0.1],
            "max_depth": [3, 4, 6, 8],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0.0, 1e-4, 1e-3, 1e-2],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
            "min_child_weight": [1, 5, 10]
        }
        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=space,
            n_iter=N_ITER,
            cv=ps,
            scoring="neg_mean_squared_error",
            random_state=self.seed,
            refit=False
        )
        search.fit(X_all, y_all)
        self.params = search.best_params_
        self.logger.info(f"[XGB] tuned params -> {self.params}")

    def train(self, X_train, y_train, X_val, y_val) -> None:
        """
        Tune on (train,val), pick best_n via early stopping on val, then refit on train+val
        """
        self._tune(X_train, y_train, X_val, y_val)

        # Early stopping to pick best_iteration
        tmp = XGBRegressor(
            **self.params,
            n_estimators=N_ESTIMATORS,
            objective="reg:squarederror",
            random_state=self.seed,
            tree_method="hist",
        )
        tmp.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="rmse",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        best_n = int(getattr(tmp, "best_iteration", None) or N_ESTIMATORS)

        # Final fit on train+val
        X_all = pd.concat([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0)
        self.model = XGBRegressor(
            **self.params,
            n_estimators=best_n,
            objective="reg:squarederror",
            random_state=self.seed,
            tree_method="hist")
        self.model.fit(X_all, y_all)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def save(self, filepath: str) -> None:
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "XGBoostModel":
        return joblib.load(filepath)
