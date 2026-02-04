import pickle
import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.base_model import JSLModel

class RFModel(JSLModel):
    """
    Implements a Random Forest classifier for identifying static signs.
    Suitable for recognizing signs that can be understood from a single photograph.
    """
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get_setting('paths', 'dataset_path')
        self.save_path = config.get_setting('paths', 'rf_model_path')

    def train(self):
        """Loads labeled movement data and trains the decision-tree based model."""
        try:
            with open(self.data_path, "rb") as f:
                dataset = pickle.load(f)
            data = np.asarray(dataset["data"])
            labels = np.asarray(dataset["labels"])
        except FileNotFoundError:
            print("Required dataset not found for training.")
            return

        # Split data into training and validation sets
        train_x, val_x, train_y, val_y = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels
        )

        print("Teaching the model to recognize signs...")
        self.model = RandomForestClassifier()
        self.model.fit(train_x, train_y)

        # Check the model's performance
        guesses = self.model.predict(val_x)
        score = accuracy_score(val_y, guesses)
        print(f"Success Rate: {score * 100:.2f}%")

    def predict(self, pattern):
        """Standardizes input shape and provides a prediction for the seen gesture."""
        if self.model is None:
            raise ValueError("Model is not ready. Please train or load it first.")
        
        sample = np.asarray(pattern)
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        return self.model.predict(sample)[0]

    def save(self, filepath=None):
        """Preserves the trained model state to the specified location."""
        dest = filepath or self.save_path
        print(f"Permanent record saved to {dest}")
        with open(dest, "wb") as f:
            pickle.dump({"model": self.model}, f)

    def load(self, filepath=None):
        """Reloads the model from storage to enable immediate use."""
        src = filepath or self.save_path
        print(f"Restoring knowledge from {src}")
        with open(src, "rb") as f:
            content = pickle.load(f)
            self.model = content["model"]
