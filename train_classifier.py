import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from constants import DATASET_PATH, MODEL_PATH


def train_model() -> None:
    """
    Loads the processed dataset, trains a RandomForestClassifier,
    evaluates its accuracy, and saves the trained model.
    """
    try:
        # 'rb' means 'read binary' mode
        with open(DATASET_PATH, "rb") as f:
            data_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: '{DATASET_PATH}' not found. Please create the dataset first.")
        return

    # Convert the data and labels lists into NumPy arrays for scikit-learn
    data = np.asarray(data_dict["data"])
    labels = np.asarray(data_dict["labels"])

    # Split the dataset into training and testing sets.
    # - test_size=0.2: 20% of the data will be used for testing, 80% for training.
    # - shuffle=True: Randomizes the order of the data before splitting.
    # - stratify=labels: Ensures that the train and test sets have the
    #   same proportion of examples from each class as the original dataset.
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, shuffle=True, stratify=labels
    )

    # Initialize the RandomForestClassifier.
    # This is a good, general-purpose model for this kind of tabular data.
    model = RandomForestClassifier()

    print("Training model...")
    # Train the model using the training data
    model.fit(x_train, y_train)

    # Use the trained model to make predictions on the *test* data
    y_predict = model.predict(x_test)

    # Compare the model's predictions (y_predict) with the true labels (y_test)
    score = accuracy_score(y_predict, y_test)

    # Print the accuracy as a percentage
    print(f"Accuracy: {score * 100:.2f}%")

    print(f"Saving model to {MODEL_PATH}...")
    # Save the trained model to a pickle file for use in inference.py
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model}, f)

    print("Model saved successfully!")