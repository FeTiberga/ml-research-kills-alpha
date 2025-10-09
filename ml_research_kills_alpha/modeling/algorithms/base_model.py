# Base class for all model implementations.
# Defines common interface for training, prediction, saving, and loading.

import abc
import pandas as pd
import numpy as np

from ml_research_kills_alpha.support import Logger
from ml_research_kills_alpha.support.constants import RANDOM_SEED


SUB_SAMPLE_PCT = 0.2


class Modeler(abc.ABC):

    def __init__(self, name: str = None):
        self.name = name
        self.logger = Logger()
        self.is_deep = False
        

    def get_subsample(self,
                      X_train: pd.DataFrame | np.ndarray, y_train: pd.Series | np.ndarray,
                      X_val: pd.DataFrame | np.ndarray, y_val: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a label-aware subsample for fast hyperparameter search.
        Combines train and validation then samples rows while preserving the target
        distribution by stratifying on y-quantile bins.

        Args:
            X_train: Training features (2D).
            y_train: Training target (1D).
            X_val: Validation features (2D).
            y_val: Validation target (1D).
            random_state: Random seed.

        Returns:
            (X_sub, y_sub): NumPy arrays for the subsample.
        """
        X_all = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)], axis=0)
        y_all = pd.concat([pd.Series(y_train).reset_index(drop=True),
                           pd.Series(y_val).reset_index(drop=True)], axis=0).reset_index(drop=True)
        # Bin y into deciles for stratification; fallback to simple sample if it fails.
        try:
            bins = pd.qcut(y_all, q=10, labels=False, duplicates="drop")
            subsample_indices = (
                X_all.assign(_bin=bins)
                    .groupby("_bin", group_keys=False)
                    .apply(lambda d: d.sample(frac=SUB_SAMPLE_PCT, random_state=RANDOM_SEED))
                    .index
            )
        except Exception:
            subsample_indices = X_all.sample(frac=SUB_SAMPLE_PCT, random_state=RANDOM_SEED).index

        X_sub = X_all.loc[subsample_indices].to_numpy()
        y_sub = y_all.loc[subsample_indices].to_numpy()
        return X_sub, y_sub

    @abc.abstractmethod
    def train(self, X_train: pd.Series, y_train: pd.Series, 
                    X_val: pd.Series, y_val: pd.Series):
        """
        Train the model on given training data, optionally using validation data for tuning.
        
        Args:
            X_train (pd.Series): Features for training.
            y_train (pd.Series): Target variable for training.
            X_val (pd.Series): Features for validation.
            y_val (pd.Series): Target variable for validation.
        """
        pass

    @abc.abstractmethod
    def predict(self, X: pd.Series) -> pd.Series:
        """
        Generate predictions for the given input data.
        
        Args:
            X (pd.Series): Input features for prediction.
            
        Returns:
            pd.Series: Predicted values.
        """
        pass
    
    @abc.abstractmethod
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """
        Evaluate the model on the given input data and return a performance metric.
        
        Args:
            y_true (pd.Series): True target values for evaluation.
            y_pred (pd.Series): Predicted values for evaluation.

        Returns:
            float: Performance metric (e.g., MAE, RMSE).
        """
        pass
    


    @abc.abstractmethod
    def save(self, filepath: str):
        """Save the model to the given file path."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath: str) -> "Modeler":
        """Load a model from the given file path."""
        pass
