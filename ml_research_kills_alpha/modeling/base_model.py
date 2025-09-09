# Base class for all model implementations.
# Defines common interface for training, prediction, saving, and loading.

import abc
import pandas as pd


class Modeler(abc.ABC):

    def __init__(self, name=None):
        self.name = name

    @abc.abstractmethod
    def train(self, X_train: pd.Series, y_train: pd.Series, 
                    X_val: pd.Series = None, y_val: pd.Series = None):
        """
        Train the model on given training data, optionally using validation data for tuning.
        
        Args:
            X_train (pd.Series): Features for training.
            y_train (pd.Series): Target variable for training.
            X_val (pd.Series): Optional features for validation.
            y_val (pd.Series): Optional target variable for validation.
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
    def save(self, filepath: str):
        """Save the model to the given file path."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath: str) -> "Modeler":
        """Load a model from the given file path."""
        pass
