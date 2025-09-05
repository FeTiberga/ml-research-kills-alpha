import pandas as pd
from ml_research_kills_alpha.modeling.base_model import Modeler
from ml_research_kills_alpha.modeling.ensemble import EnsembleModel

class RollingTrainer:
    """Performs expanding-window training from start_year to end_year with a 6-year validation period before each test year."""
    def __init__(self, models: list[Modeler], data, start_year=2005, end_year=2021, target_col='target'):
        """
        models: list of Modeler instances to train.
        data: pandas DataFrame with features, target, and a 'year' column for time-based slicing.
        """
        self.models = models
        self.data = data.copy()
        self.start_year = start_year
        self.end_year = end_year
        self.target_col = target_col
        # Ensure 'year' column exists in data for slicing
        if 'year' not in self.data.columns:
            raise ValueError("Data must contain a 'year' column for time-based slicing.")

    def run(self):
        results = {}
        for year in range(self.start_year, self.end_year + 1):
            print(f"Training models for test year {year}...")
            # Define training, validation, testing periods
            train_end_year = year - 7
            val_start_year = year - 6
            val_end_year = year - 1
            # Split data into training, validation, and test sets
            train_data = self.data[self.data['year'] <= train_end_year]
            val_data = self.data[(self.data['year'] >= val_start_year) & (self.data['year'] <= val_end_year)]
            test_data = self.data[self.data['year'] == year]
            # Separate features (drop target and year) and target
            X_train = train_data.drop(columns=[self.target_col, 'year'], errors='ignore')
            y_train = train_data[self.target_col]
            X_val = val_data.drop(columns=[self.target_col, 'year'], errors='ignore')
            y_val = val_data[self.target_col]
            X_test = test_data.drop(columns=[self.target_col, 'year'], errors='ignore')
            y_test = test_data[self.target_col] if self.target_col in test_data.columns else None
            results[year] = {}
            # Train each model and record test set predictions
            for model in self.models:
                model.train(X_train, y_train, X_val, y_val)
                preds = model.predict(X_test)
                results[year][model.name] = preds
            # Ensemble of all deep learning models (FFNNs and LSTMs)
            deep_models = [m for m in self.models if getattr(m, 'is_deep', False)]
            if deep_models:
                ensemble_model = EnsembleModel(deep_models)
                ensemble_preds = ensemble_model.predict(X_test)
                results[year][ensemble_model.name] = ensemble_preds
        return results
