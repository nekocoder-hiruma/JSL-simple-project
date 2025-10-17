import pickle
from typing import Tuple, Optional, Any

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from constants import DATASET_PATH, MODEL_PATH, NUMBER_OF_CLASSES, SEQUENCE_LENGTH


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
        label_array = np.asarray(data_dict["labels"])
        # One-hot encode labels for categorical_crossentropy
        labels = to_categorical(label_array, num_classes=NUMBER_OF_CLASSES).astype(int)
        return data, labels
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Please create the dataset first.")
        return None

def prepare_data(data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
np.ndarray, np.ndarray]:
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


def build_lstm_model() -> Sequential:
    """
    Builds, compiles, and returns the LSTM model.
    """
    model = Sequential([
        # *** FIX 1: Imports for these layers are now added ***
        # The input_shape is (timesteps, features)
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 42)),
        Dropout(0.2),  # Dropout helps prevent overfitting
        LSTM(64, return_sequences=False),  # The last LSTM layer doesn't return a sequence
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(NUMBER_OF_CLASSES, activation='softmax')  # Output layer
    ])

    # Compile the model with an optimizer, loss function, and metrics
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # Print a summary of the model architecture
    return model


def train_and_save_model(model: Sequential, x_train, y_train, x_test, y_test) -> None:
    """
    Trains the model and saves it to a file.
    """
    print("\n--- Starting Model Training ---")
    # Train the model
    model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

    print("\n--- Training Complete. Saving model... ---")
    # Save the trained Keras model
    model.save(MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")


def main_trainer() -> None:
    """
    Main orchestrator function for the training pipeline.
    """
    dataset = load_dataset(DATASET_PATH)
    if dataset is None:
        return

    data, labels = dataset
    x_train, x_test, y_train, y_test = prepare_data(data, labels)

    model = build_lstm_model()

    train_and_save_model(model, x_train, y_train, x_test, y_test)
