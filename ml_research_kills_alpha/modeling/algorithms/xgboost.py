import joblib
import xgboost as xgb
from .base_model import Modeler

class XGBoostModel(Modeler):
    """XGBoost regression model for return prediction."""
    def __init__(self, params=None):
        # Default parameters for XGBRegressor if not provided
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': 1000,
            'learning_rate': 0.1
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None
        self.name = "XGBoost"
        self.is_deep = False

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Initialize XGBRegressor with given parameters
        self.model = xgb.XGBRegressor(**self.params)
        if X_val is not None and y_val is not None:
            # Use early stopping with validation data (10 rounds patience)
            self.model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)],
                           early_stopping_rounds=10,
                           verbose=False)
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)
