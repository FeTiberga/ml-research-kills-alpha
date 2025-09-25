import pandas as pd
import numpy as np
from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.modeling.algorithms.ensemble import EnsembleModel
from ml_research_kills_alpha.datasets.useful_files import CZ_SIGNAL_DOC
from ml_research_kills_alpha.support.constants import NON_FEATURES, PREDICTED_COL
from ml_research_kills_alpha.support import Logger

VALIDATION_PERIOD = 6
START_YEAR = 2005
CZ_DF = pd.read_csv(CZ_SIGNAL_DOC)
CZ_YEAR_COLUMN = "Year"
CZ_FEATURE_COLUMN = "Acronym"


class RollingTrainer:
    """
    Performs expanding-window training from start_year to end_year with a 
    {VALIDATION_PERIOD}-year validation period before each test year.
    """
    def __init__(self, models: list[Modeler], data: pd.DataFrame,
                 end_year: int = 2021, target_col: str = 'ret'):
        """
        Args:
            models (list[Modeler]): list of Modeler instances to train.
            data (pd.DataFrame): financial data with features, return, and a 'year' column for time-based slicing.
            start_year (int): first test year (inclusive).
            end_year (int): last test year (inclusive).
            target_col (str): name of the target column to predict.
        """
        self.models = models
        self.data = data.copy()
        self.start_year = START_YEAR
        self.end_year = end_year
        self.target_col = target_col
        self.year_col = 'date'
        
        self.logger = Logger()
        self.logger.info(f"RollingTrainer initialized for years {self.start_year} to {self.end_year} with target column: '{self.target_col}'")

        # Ensure 'date' column exists in data for slicing
        if self.year_col not in self.data.columns:
            self.logger.error(f"Data must contain a '{self.year_col}' column for time-based slicing.")
            raise ValueError(f"Data must contain a '{self.year_col}' column for time-based slicing.")

    def filter_features(self, year: int) -> pd.DataFrame:
        """
        Filter the features we can use on a specific year based on
        when the signals was first published.
        
        Args:
            year (int): the year to filter features for.
        """
        valid_features = CZ_DF[CZ_DF[CZ_YEAR_COLUMN] <= year][CZ_FEATURE_COLUMN].tolist()
        all_features = CZ_DF[CZ_FEATURE_COLUMN].tolist()
        # only consider features that are in the data
        valid_features = [f for f in valid_features if f in self.data.columns]
        total_features = len([col for col in all_features if col in self.data.columns])
        self.logger.info(f"Year {year}: Using {len(valid_features)} valid features out of {total_features} total features.")
        return self.data[valid_features + [self.year_col] + [self.target_col]]
    
    def train_val_test_split(self, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets based on the given year.
        
        Args:
            year (int): the test year.
        
        Returns:
            train_data (pd.DataFrame): data for training.
            val_data (pd.DataFrame): data for validation.
            test_data (pd.DataFrame): data for testing.
        """
        train_end_year = year - VALIDATION_PERIOD - 1
        val_start_year = year - VALIDATION_PERIOD
        val_end_year = year - 1
        
        features_data = self.filter_features(year)
        features_data[self.year_col] = pd.to_datetime(features_data[self.year_col])

        train_data = features_data[features_data[self.year_col].dt.year <= train_end_year]
        val_data = features_data[(features_data[self.year_col].dt.year >= val_start_year) & (features_data[self.year_col].dt.year <= val_end_year)]
        test_data = features_data[features_data[self.year_col].dt.year == year]

        self.logger.info(f"Year {year}: Train data from start to {train_end_year}, Validation data from {val_start_year} to {val_end_year}, Test data for {year}.")
        return train_data, val_data, test_data
    
    @staticmethod
    def get_x(df: pd.DataFrame) -> pd.Series:
        """
        Get the feature matrix X by excluding non-feature columns.

        Returns:
            pd.Series: Feature matrix.
        """
        features = [col for col in df.columns if col not in NON_FEATURES + [PREDICTED_COL]]
        return df[features]

    def get_y(self, df: pd.DataFrame) -> pd.Series:
        """
        Get the target vector y by excluding non-feature columns.

        Returns:
            pd.Series: Target vector.
        """
        return df[self.target_col]

    def run(self):
        results = {}
        for year in range(self.start_year, self.end_year + 1):
            self.logger.info(f"Training models for test year {year}...")

            # Split data into training, validation, and test sets
            train_data, val_data, test_data = self.train_val_test_split(year)
            
            # prepare data for modeling
            X_train = self.get_x(train_data)
            y_train = self.get_y(train_data)
            X_val = self.get_x(val_data)
            y_val = self.get_y(val_data)
            X_test = self.get_x(test_data)
            y_test = self.get_y(test_data)
            results[year] = {}

            # Train each model and record test set predictions
            for model in self.models:
                self.logger.info(f"Training model: {model.name} for year {year}")
                model.train(X_train, y_train, X_val, y_val)
                loss = model.evaluate(X_test, y_test)
                results[year][model.name] = loss
    
            # Ensemble of all deep learning models (FFNNs and LSTMs)
            deep_models = [m for m in self.models if getattr(m, 'is_deep', False)]
            if deep_models:
                self.logger.info(f"Evaluating ensemble of deep models for year {year}")
                ensemble_model = EnsembleModel(deep_models)
                ensemble_loss = ensemble_model.evaluate(X_test, y_test)
                results[year][ensemble_model.name] = ensemble_loss

        return results
