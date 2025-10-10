import pandas as pd
import numpy as np
import re

from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.datasets.useful_files import CZ_SIGNAL_DOC
from ml_research_kills_alpha.support import Logger, NON_FEATURES, PREDICTED_COL, META_COLS, LSTM_SEQUENCE_LENGTH


VALIDATION_PERIOD = 6
START_YEAR = 2005
CZ_DF = pd.read_csv(CZ_SIGNAL_DOC)
CZ_YEAR_COLUMN = "Year"
CZ_FEATURE_COLUMN = "Acronym"


def to_snake_case(name):
    # Replace spaces and hyphens with underscores
    name = re.sub(r"[ -]+", "_", name)
    # Convert CamelCase to snake_case
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Lowercase everything
    return name.lower()
# Convert each value in the CZ_FEATURE_COLUMN column to snake_case
if CZ_FEATURE_COLUMN in CZ_DF.columns:
    CZ_DF[CZ_FEATURE_COLUMN] = CZ_DF[CZ_FEATURE_COLUMN].apply(to_snake_case)


class RollingTrainer:
    """
    Performs expanding-window training from start_year to end_year with a 
    {VALIDATION_PERIOD}-year validation period before each test year.
    """
    def __init__(self, models: list[Modeler], data: pd.DataFrame,
                 end_year: int = 2021, target_col: str = 'abret'):
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
        features_data.loc[:, self.year_col] = pd.to_datetime(features_data[self.year_col])

        train_data = features_data[features_data[self.year_col].dt.year <= train_end_year]
        val_data = features_data[(features_data[self.year_col].dt.year >= val_start_year) & (features_data[self.year_col].dt.year <= val_end_year)]
        test_data = features_data[features_data[self.year_col].dt.year == year]

        self.logger.info(f"Year {year}: Train data from start to {train_end_year}, Validation data from {val_start_year} to {val_end_year}, Test data for {year}.")
        return train_data, val_data, test_data
    
    def get_x(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the feature matrix X by excluding non-feature columns.

        Returns:
            pd.DataFrame: Feature matrix.
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
    
    def build_sequences(self, panel: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Build LSTM sequences grouped by permno sorted by date.

        Args:
            panel (pd.DataFrame): Long panel with ['permno','date', features..., target_col]
            feature_cols (list[str]): Feature names used as X.

        Returns:
            tuple: X_seq (N, T, D), y (N,)
        """
        panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
        X_list, y_list = [], []
        for permno, g in panel.groupby("permno", sort=False):
            if len(g) < LSTM_SEQUENCE_LENGTH + 1:
                continue
            feats = g[feature_cols].to_numpy(dtype=np.float32)
            tgt = g[self.target_col].to_numpy(dtype=np.float32)
            for t in range(LSTM_SEQUENCE_LENGTH, len(g)):
                X_list.append(feats[t - LSTM_SEQUENCE_LENGTH:t, :])
                y_list.append(tgt[t])

        if not X_list:
            D = len(feature_cols)
            return (
                np.empty((0, LSTM_SEQUENCE_LENGTH, D), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return np.stack(X_list), np.array(y_list)
    
    def _seq_indexed(self, panel: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Build sequences with an aligned index for (permno, date). Used by the LSTM.
        The 'date' in the returned index corresponds to the prediction month.

        Args:
            panel (pd.DataFrame): Long panel with ['permno','date', features..., target_col].
            feature_cols (list[str]): Features to feed the LSTM.

        Returns:
            tuple:
                - X_seq (np.ndarray): shape (N, LSTM_SEQUENCE_LENGTH, D)
                - y (np.ndarray): shape (N,)
                - idx (pd.DataFrame): rows aligned with X_seq/y; columns ['permno','date']
        """
        panel = panel.sort_values(["permno","date"]).reset_index(drop=True)
        X_list, y_list, idx_rows = [], [], []

        for permno, g in panel.groupby("permno", sort=False):
            if len(g) < LSTM_SEQUENCE_LENGTH + 1:
                continue
            feats = g[feature_cols].to_numpy(dtype=np.float32)
            tgt   = g[self.target_col].to_numpy(dtype=np.float32)
            dates = g["date"].to_numpy()

            for t in range(LSTM_SEQUENCE_LENGTH, len(g)):
                X_list.append(feats[t - LSTM_SEQUENCE_LENGTH:t, :])
                y_list.append(tgt[t])
                idx_rows.append({"permno": int(permno), "date": dates[t]})

        if not X_list:
            D = len(feature_cols)
            return (
                np.empty((0, LSTM_SEQUENCE_LENGTH, D), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                pd.DataFrame(columns=["permno","date"])
            )

        return np.stack(X_list), np.asarray(y_list), pd.DataFrame(idx_rows)

    def run_model(self, model: Modeler,
                  X_train: pd.Series, y_train: pd.Series,
                  X_val: pd.Series, y_val: pd.Series,
                  year: int, test_sorted: pd.DataFrame,
                  months: pd.Series, preds_all: list,
                  val_panel: pd.DataFrame | None = None,
                  feature_cols: list[str] | None = None
                  ) -> tuple[float, list]:
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
            val_panel (pd.DataFrame | None): full validation panel for LSTM sequence building.
            feature_cols (list[str]): feature columns used for LSTM sequence building.
        """
        self.logger.info(" ")
        self.logger.info(f"Training model: {model.name} for year {year}")
        model.train(X_train, y_train, X_val, y_val)

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
            
            # LSTM needs sequences built from (val + test up to curr_month)
            if model.name.startswith("LSTM"):
                if val_panel is None or feature_cols is None:
                    self.logger.error("val_panel and feature_cols must be provided for LSTM models.")
                    raise ValueError("val_panel and feature_cols must be provided for LSTM models.")
                # Build sequences from (val_panel + test rows up to curr_month)
                history = pd.concat([val_panel, test_sorted[test_sorted[self.year_col] <= curr_df[self.year_col].max()]])
                X_seq, _, idx = self._seq_indexed(history, feature_cols)
                if X_seq.shape[0] == 0 or idx.empty:
                    self.logger.warning(f"No sequences could be built for LSTM model {model.name} for month {curr_month}. Skipping.")
                    continue
                # select sequences whose prediction month equals prev_month
                mask = (idx["date"].astype("period[M]") == prev_month)
                if not mask.any():
                    self.logger.warning(f"No sequences found for LSTM model {model.name} for month {curr_month}. Skipping.")
                    continue
                yhat = model.predict(X_seq[mask.values])
                prev_preds = idx.loc[mask, ["permno", "date"]].copy()
                prev_preds["model"] = model.name
                prev_preds["y_hat"] = yhat
        
            # Other models use get_x directly
            else:
                yhat = model.predict(self.get_x(prev_df))
                prev_preds = prev_df[["permno", self.year_col]].copy()
                prev_preds["y_hat"] = np.asarray(yhat)
                prev_preds["model"] = model.name
                prev_preds["date"]  = prev_preds[self.year_col].dt.to_period("M")
                        
            preds_all.append(prev_preds[['permno','date','model','y_hat']])

            # Proper loss: align by permno intersection
            merged = curr_df[['permno', self.target_col]].merge(
                prev_preds[['permno','y_hat']], on='permno', how='inner'
            )
            if not merged.empty:
                loss = model.evaluate(merged[self.target_col].values, merged['y_hat'].values)
                monthly_losses.append(loss)
                
        return (float(np.mean(monthly_losses)) if monthly_losses else float("nan")), preds_all

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
                
                # LSTM needs special sequence data
                if model.name.startswith("LSTM"):
                    # feature_cols from *panel* that still has meta cols
                    exclude = set(NON_FEATURES) | set(META_COLS) | set(PREDICTED_COL) | {self.target_col}
                    feature_cols = [c for c in train_data.columns if c not in exclude]
                    X_train_lstm, y_train_lstm = self.build_sequences(train_data, feature_cols)
                    X_val_lstm, y_val_lstm = self.build_sequences(val_data, feature_cols)
                    if X_train_lstm.shape[0] == 0 or X_val_lstm.shape[0] == 0:
                        self.logger.warning(f"Not enough sequences to train LSTM model {model.name} for year {year}. Skipping.")
                        continue
                    avg_loss, preds_all = self.run_model(model,
                                                         X_train_lstm, y_train_lstm,
                                                         X_val_lstm, y_val_lstm,
                                                         year, test_sorted, months, preds_all,
                                                         val_panel=val_data,
                                                         feature_cols=feature_cols)
                    results.setdefault(year, {})[model.name] = avg_loss
                    
                # Other models work with X, y directly
                else:
                    avg_loss, preds_all = self.run_model(model, X_train, y_train, X_val, y_val, year, test_sorted, months, preds_all)
                    results.setdefault(year, {})[model.name] = avg_loss

        preds_all = pd.concat(preds_all, axis=0).reset_index(drop=True)
        return preds_all, results
