import os
import json
from ml_research_kills_alpha.modeling.base_model import Modeler
from ml_research_kills_alpha.modeling.elastic_net import ElasticNetModel
from ml_research_kills_alpha.modeling.huber_ols import HuberRegressorModel
from ml_research_kills_alpha.modeling.xgboost import XGBoostModel
from ml_research_kills_alpha.modeling.neural_networks import FFNNModel
from ml_research_kills_alpha.modeling.lstm import LSTMModel

from .neural_networks import FFNNModel, LSTMModel

class EnsembleModel(Modeler):
    """Ensemble model that averages predictions from multiple trained models (e.g., FFNNs and LSTMs)."""
    def __init__(self, models_list):
        self.models = models_list  # list of Modeler instances
        self.name = "Ensemble"
        # If any sub-model is deep, mark ensemble as deep strategy
        self.is_deep = any(getattr(m, 'is_deep', False) for m in models_list)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # No training needed for ensemble (assumes sub-models are already trained)
        return

    def predict(self, X):
        import numpy as np
        # Average predictions from all sub-models
        preds_all = []
        for model in self.models:
            preds = model.predict(X)
            preds_all.append(preds)
        preds_all = np.array(preds_all)
        return preds_all.mean(axis=0)

    def save(self, filepath):
        # Save each sub-model into the specified directory and record their class and file
        os.makedirs(filepath, exist_ok=True)
        model_files = []
        for idx, model in enumerate(self.models):
            model_filename = f"model_{idx}.pkl"
            model_path = os.path.join(filepath, model_filename)
            model.save(model_path)
            model_files.append((model.__class__.__name__, model_filename))
        # Save manifest with class names and file names
        manifest = {"models": model_files}
        with open(os.path.join(filepath, "manifest.json"), "w") as f:
            json.dump(manifest, f)

    @classmethod
    def load(cls, filepath):
        manifest_path = os.path.join(filepath, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        models_list = []
        # Map class names to actual classes for loading
        class_map = {
            "ElasticNetModel": ElasticNetModel,
            "HuberRegressorModel": HuberRegressorModel,
            "XGBoostModel": XGBoostModel,
            "FFNNModel": FFNNModel,
            "LSTMModel": LSTMModel,
            "EnsembleModel": cls
        }
        for class_name, file_name in manifest["models"]:
            if class_name not in class_map:
                raise ValueError(f"Unknown model class {class_name} in ensemble manifest.")
            model_cls = class_map[class_name]
            model = model_cls.load(os.path.join(filepath, file_name))
            models_list.append(model)
        return cls(models_list)
