import pickle
import numpy as np
from typing import Optional
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from models.base_model import JSLModel

class LSTMModel(JSLModel):
    """
    Implements a sequence-based classifier for recognizing motion-based signs.
    Analyzes how hand positions change over time to identify dynamic gestures.
    """
    def __init__(self, config):
        super().__init__(config)
        self.data_path = config.get_setting('paths', 'dataset_path')
        self.save_path = config.get_setting('paths', 'keras_model_path')
        self.num_classes = config.get_int('model_params', 'number_of_classes', 0)
        self.seq_len = config.get_int('model_params', 'sequence_length', 30)

    def train(self):
        """Loads motion sequences and teaches the layered model to recognize patterns."""
        try:
            with open(self.data_path, "rb") as f:
                dataset = pickle.load(f)
            data = np.asarray(dataset["data"])
            raw_labels = np.asarray(dataset["labels"])
            # Format labels for categorical recognition
            labels = to_categorical(raw_labels, num_classes=self.num_classes).astype(int)
        except FileNotFoundError:
            print("Required dataset not found for training.")
            return

        train_x, val_x, train_y, val_y = train_test_split(
            data, labels, test_size=0.2, shuffle=True, stratify=labels
        )

        print("Building the neural framework for movement analysis...")
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.seq_len, 42)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        print("\n--- Model Training in Progress ---")
        self.model.fit(train_x, train_y, epochs=80, validation_data=(val_x, val_y))

    def predict(self, motion_sequence) -> tuple[int, float]:
        """Analyzes movement and returns the most likely sign with a confidence score."""
        if self.model is None:
            raise ValueError("Model is not ready. Please train or load it first.")
        
        # Prepare the input shape for the model
        sample = np.asarray(motion_sequence)
        if sample.ndim == 2:
            sample = np.expand_dims(sample, axis=0)
            
        # Get raw probabilities from the softmax output
        predictions = self.model.predict(sample)[0]
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])
        
        return class_idx, confidence

    def save(self, filepath=None):
        """Saves the model's internal structure and weights to a file."""
        dest = filepath or self.save_path
        print(f"Model progress saved to {dest}")
        self.model.save(dest)

    def load(self, filepath=None):
        """Restores a previously trained model for immediate use."""
        src = filepath or self.save_path
        print(filepath, self.save_path)
        print(f"Loading model state from {src}")
        self.model = load_model(src)
