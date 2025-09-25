# Base class for all model implementations.
# Defines common interface for training, prediction, saving, and loading.

import abc
import pandas as pd
import numpy as np

from ml_research_kills_alpha.support import Logger
from ml_research_kills_alpha.support.constants import RANDOM_SEED

class Modeler(abc.ABC):

    def __init__(self, name: str = None):
        self.name = name
        self.logger = Logger()
        
    @abc.abstractmethod
    def get_subsample(self,
                      sub_sample_pct: float,
                      X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      random_state: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
        """
        Fuse train and val, then return a stratified 20% representative subsample.
        """
        # Concatenate train and val
        X_all = pd.concat([X_train, X_val], axis=0)
        y_all = pd.concat([y_train, y_val], axis=0)

        # Stratified sampling
        subsample_indices = X_all.sample(frac=sub_sample_pct, random_state=random_state).index

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
    def evaluate(self, X: pd.Series, y: pd.Series) -> float:
        """
        Evaluate the model on the given input data and return a performance metric.
        
        Args:
            X (pd.Series): Input features for prediction.
            y (pd.Series): True target values for evaluation.
            
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
