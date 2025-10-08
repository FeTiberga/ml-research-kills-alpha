import pandas as pd
import numpy as np
from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.modeling.algorithms.ensemble import EnsembleModel
from ml_research_kills_alpha.datasets.useful_files import CZ_SIGNAL_DOC
from ml_research_kills_alpha.support.constants import NON_FEATURES, PREDICTED_COL, META_COLS
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
        
        self.prepare_data()
        
    def prepare_data(self):
        """
        Prepare the data by ensuring correct dtypes and adding necessary columns.
        """
        self.logger.info(" ")
        self.logger.info(f"Preparing data, original shape: {self.data.shape}")
        
        # Ensure 'date' is datetime
        self.data[self.year_col] = pd.to_datetime(self.data[self.year_col])
        if self.target_col not in self.data.columns:
            self.logger.error(f"Target column '{self.target_col}' not found in data.")
            raise ValueError(f"Target column '{self.target_col}' not found in data.")
    
        # drop rows with NaN in target column
        self.data = self.data.dropna(subset=[self.target_col])
        self.data = self.data.sort_values(by=[self.year_col, 'permno']).reset_index(drop=True)
        self.logger.info(f"Data preparation complete. Data shape: {self.data.shape}")

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
        meta_cols = [col for col in META_COLS if col in self.data.columns]
        total_features = len([col for col in all_features if col in self.data.columns])
        self.logger.info(f"Year {year}: Using {len(valid_features)} valid features out of {total_features} total features.")
        return self.data[valid_features + meta_cols + [self.year_col, self.target_col]]

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
    
    def get_x(self, df: pd.DataFrame) -> pd.Series:
        """
        Get the feature matrix X by excluding non-feature columns.

        Returns:
            pd.Series: Feature matrix.
        """ 
        exclude = set(NON_FEATURES) | set(META_COLS) | set(PREDICTED_COL) | {self.target_col}
        features = [col for col in df.columns if col not in exclude]
        return df[features]

    def get_y(self, df: pd.DataFrame) -> pd.Series:
        """
        Get the target vector y by excluding non-feature columns.

        Returns:
            pd.Series: Target vector.
        """
        return df[self.target_col]
    
    def run_model(self, model: Modeler,
                  X_train: pd.Series, y_train: pd.Series,
                  X_val: pd.Series, y_val: pd.Series,
                  year: int, test_sorted: pd.DataFrame,
                  months: pd.Series, preds_all: list,
                  train: bool = True) -> tuple[float, list]:
        """
        Train and evaluate a single model for a specific year.
        
        Args:
            model (Modeler): the model to train and evaluate.
            X_train (pd.Series): training features.
            y_train (pd.Series): training target.
            X_val (pd.Series): validation features.
            y_val (pd.Series): validation target.
            year (int): the test year.
            test_sorted (pd.DataFrame): sorted test data for evaluation.
            months (pd.Series): unique months in the test data.
            preds_all (list): list to append predictions to.
            train (bool): whether to train the model or not - for ensembles is False.
        """
        self.logger.info(" ")
        if train:
            self.logger.info(f"Training model: {model.name} for year {year}")
            model.train(X_train, y_train, X_val, y_val)
        else:   
            self.logger.info(f"Using pre-trained model: {model.name} for year {year}")

        monthly_losses = []

        for i in range(1, len(months)):
            self.logger.info(f"Evaluating model: {model.name} for year {year}, month {months[i]}")
            prev_month = months[i - 1] 
            curr_month = months[i]
            prev_df: pd.DataFrame = test_sorted[test_sorted['formation_month'] == prev_month].copy()
            curr_df: pd.DataFrame = test_sorted[test_sorted['formation_month'] == curr_month].copy()

            if prev_df.empty or curr_df.empty:
                continue

            # drop formation_month column
            prev_df = prev_df.drop(columns=['formation_month'])
            curr_df = curr_df.drop(columns=['formation_month'])

            # Predict on prev cross-section
            yhat = model.predict(self.get_x(prev_df))
            prev_preds = prev_df[['permno', self.year_col]].copy()
            prev_preds['y_hat'] = np.asarray(yhat)
            prev_preds['model'] = model.name
            # mark formation month as the 'date' you'll use for portfolio formation
            prev_preds['date'] = prev_preds[self.year_col].dt.to_period('M')
            
            preds_all.append(prev_preds[['permno','date','model','y_hat']])

            # Proper loss: align by permno intersection
            merged = curr_df[['permno', self.target_col]].merge(
                prev_preds[['permno','y_hat']], on='permno', how='inner'
            )
            if not merged.empty:
                loss = model.evaluate(merged[self.target_col].values, merged['y_hat'].values)
                monthly_losses.append(loss)
                
        return np.mean(monthly_losses) if monthly_losses else None, preds_all

    def run(self):
        """
        Run the rolling training process.
        
        Returns:
            preds_all (pd.DataFrame): DataFrame containing predictions for each model and year.
            results (dict): Dictionary containing evaluation metrics for each model and year.
        """
        preds_all = []
        results = {}
        
        for year in range(self.start_year, self.end_year + 1):
            self.logger.info(f"Training models for test year {year}...")

            # Split data into training, validation, and test sets
            train_data, val_data, test_data = self.train_val_test_split(year)

            # Prepare data for modeling
            X_train = self.get_x(train_data)
            y_train = self.get_y(train_data)
            X_val = self.get_x(val_data)
            y_val = self.get_y(val_data)
            
            test_sorted = test_data.sort_values(self.year_col)
            # append the last month of the previous year val set
            last_val_month = val_data[self.year_col].dt.to_period('M').max()
            last_val_df = val_data[val_data[self.year_col].dt.to_period('M') == last_val_month].copy()
            test_sorted = pd.concat([last_val_df, test_sorted], axis=0).reset_index(drop=True)
            
            test_sorted['formation_month'] = test_sorted[self.year_col].dt.to_period('M')
            months = test_sorted['formation_month'].sort_values().unique()

            results[year] = {}

            # Train each model
            for model in self.models:
                avg_loss, preds_all = self.run_model(model, X_train, y_train, X_val, y_val, year, test_sorted, months, preds_all)
                results.setdefault(year, {})[model.name] = avg_loss

            # Ensemble of all deep learning models (FFNNs and LSTMs)
            deep_models = [m for m in self.models if getattr(m, 'is_deep', False)]
            if deep_models:
                ensemble_model = EnsembleModel(deep_models)
                avg_loss, preds_all = self.run_model(ensemble_model, X_train, y_train, X_val, y_val, year, test_sorted, months, preds_all,
                                                     train=False)
                results[year]['Ensemble'] = avg_loss

        preds_all = pd.concat(preds_all, axis=0).reset_index(drop=True)
        return preds_all, results
