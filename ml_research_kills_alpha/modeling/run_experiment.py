import pandas as pd
from ml_research_kills_alpha.modeling.elastic_net import ElasticNetModel
from ml_research_kills_alpha.modeling.huber_ols import HuberRegressorModel
from ml_research_kills_alpha.modeling.xgboost import XGBoostModel
from ml_research_kills_alpha.modeling.neural_networks import FFNNModel
from ml_research_kills_alpha.modeling.lstm import LSTMModel
from ml_research_kills_alpha.modeling.rolling_trainer import RollingTrainer

# Load the cleaned data (panel format with 'year' column, firm features, macro predictors, and target)
data = pd.read_csv("cleaned_data.csv")  # Replace with the actual path to the cleaned dataset

# Initialize all model instances
models = [
    ElasticNetModel(),
    HuberRegressorModel(),
    XGBoostModel(),
    FFNNModel(num_layers=2),
    FFNNModel(num_layers=3),
    FFNNModel(num_layers=4),
    FFNNModel(num_layers=5),
    LSTMModel(num_layers=1),
    LSTMModel(num_layers=2)
]

# Set up rolling trainer for expanding-window training from 2005 to 2021
trainer = RollingTrainer(models, data, start_year=2005, end_year=2021, target_col='target')

# Run training and get out-of-sample predictions for each year and model
results = trainer.run()

# `results` is a dictionary with structure: results[year][model_name] = predictions for that year's test set.
# You can add code here to evaluate performance (e.g., calculate returns, Sharpe ratios) or save the results.
