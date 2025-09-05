import abc


class Modeler(abc.ABC):
    """Abstract base class for all model implementations."""

    def __init__(self, name=None):
        self.name = name

    @abc.abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model on given training data, optionally using validation data for tuning."""
        pass

    @abc.abstractmethod
    def predict(self, X):
        """Generate predictions for the given input data."""
        pass

    @abc.abstractmethod
    def save(self, filepath):
        """Save the model to the given file path."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, filepath):
        """Load a model from the given file path."""
        pass
