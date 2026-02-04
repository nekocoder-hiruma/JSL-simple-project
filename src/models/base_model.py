from abc import ABC, abstractmethod
from typing import Any, Optional
from core.config_loader import ConfigLoader

class JSLModel(ABC):
    """
    Blueprint for all sign language classification models.
    Defines the standard workflow for training, predicting, and saving model progress.
    """
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.model = None

    @abstractmethod
    def train(self) -> None:
        """Starts the learning process using the prepared dataset."""
        pass

    @abstractmethod
    def predict(self, data: Any) -> tuple[int, float]:
        """Guesses the sign and provides a confidence score (0.0 to 1.0)."""
        pass

    @abstractmethod
    def save(self, filepath: Optional[str] = None) -> None:
        """Stores the model's knowledge to a file for later use."""
        pass

    @abstractmethod
    def load(self, filepath: Optional[str] = None) -> None:
        """Retrieves a previously saved model state."""
        pass
