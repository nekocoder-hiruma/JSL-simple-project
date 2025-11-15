import pickle
from typing import Tuple, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config_loader import DATASET_PATH, MODEL_PATH


def load_dataset(filepath: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads the dataset from a pickle file.

    Args:
        filepath (str): Path to the data.pickle file.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: A tuple of (data, labels)
                                                or None if the file is not found.
    """
    try:
        with open(filepath, "rb") as f:
            data_dict = pickle.load(f)
        data = np.asarray(data_dict["data"])
        labels = np.asarray(data_dict["labels"])
        return data, labels
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Please create the dataset first.")
        return None

def prepare_data(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training and testing sets.

    Args:
        data (np.ndarray): The full feature dataset.
        labels (np.ndarray): The full labels dataset.

    Returns:
        Tuple: (x_train, x_test, y_train, y_test)
    """
    return train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

def train_classifier(x_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Initializes and trains a RandomForestClassifier.

    Args:
        x_train (np.ndarray): Training utils.
        y_train (np.ndarray): Training labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    print("Training model...")
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model: RandomForestClassifier, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluates the model on the test set and prints the accuracy.

    Args:
        model (RandomForestClassifier): The trained model.
        x_test (np.ndarray): Test utils.
        y_test (np.ndarray): Test labels.
    """
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f"Accuracy: {score * 100:.2f}%")

def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """
    Saves the trained model to a pickle file.

    Args:
        model (RandomForestClassifier): The trained model.
        filepath (str): The path to save the model.
    """
    print(f"Saving model to {filepath}...")
    try:
        with open(filepath, "wb") as f:
            pickle.dump({"model": model}, f)
        print("Model saved successfully!")
    except IOError as e:
        print(f"Error saving model to {filepath}: {e}")

def train_model() -> None:
    """
    Main orchestrator function for the training pipeline.
    """
    dataset = load_dataset(DATASET_PATH)
    if dataset is None:
        return

    data, labels = dataset
    x_train, x_test, y_train, y_test = prepare_data(data, labels)
    model = train_classifier(x_train, y_train)
    evaluate_model(model, x_test, y_test)
    save_model(model, MODEL_PATH)