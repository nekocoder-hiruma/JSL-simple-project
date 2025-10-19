import os
import pickle
from typing import List, Optional

import cv2

from src.config_loader import DATA_DIR, DATASET_PATH, hands


def process_image(image_path: str) -> Optional[List[float]]:
    """
    Reads a single image, processes it with MediaPipe, and returns normalized
    hand landmarks.

    Args:
        image_path (str): The full path to the image file.

    Returns:
        Optional[List[float]]: A list of 42 normalized landmark coordinates
                               (x1_norm, y1_norm, x2_norm, y2_norm...)
                               or None if no hand is detected.
    """
    landmarks_per_image = []
    x_coords = []
    y_coords = []

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)

            # --- Normalization Step ---
            base_x, base_y = min(x_coords), min(y_coords)
            for x, y in zip(x_coords, y_coords):
                landmarks_per_image.append(x - base_x)
                landmarks_per_image.append(y - base_y)

            return landmarks_per_image
        else:
            return None

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def save_dataset(data: List[List[float]], labels: List[str], filepath: str) -> None:
    """
    Saves the collected data and labels to a pickle file.

    Args:
        data (List[List[float]]): The list of landmark data.
        labels (List[str]): The list of corresponding labels.
        filepath (str): The path to save the pickle file.
    """
    if not data:
        print("No data was processed. The dataset is empty. Did you run collection?")
        return

    print(f"Saving dataset to {filepath}...")
    try:
        with open(filepath, "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
        print("Dataset created successfully!")
    except IOError as e:
        print(f"Error saving dataset to {filepath}: {e}")


def create_dataset() -> None:
    """
    Main orchestrator function.
    Iterates through image files, processes them, and saves the resulting dataset.
    """
    data = []
    labels = []

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please run data collection first.")
        return

    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)
        if not os.path.isdir(dir_path):
            continue

        print(f"Processing class: {dir_name}")
        for img_name in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_name)

            normalized_landmarks = process_image(img_full_path)

            if normalized_landmarks:
                data.append(normalized_landmarks)
                labels.append(dir_name)

    save_dataset(data, labels, DATASET_PATH)