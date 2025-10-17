import os
import pickle
import cv2
import numpy as np
from constants import DATA_DIR, DATASET_PATH, hands


def create_dataset() -> None:
    """
    Processes all saved images in the DATA_DIR.

    This function will:
    1. Loop through each class directory (e.g., './data/0', './data/1').
    2. Loop through each image in that class directory.
    3. Use MediaPipe to detect hand landmarks in the image.
    4. **Normalize** the landmarks (a crucial step!).
    5. Store the normalized landmarks in a 'data' list.
    6. Store the corresponding class (e.g., '0', '1') in a 'labels' list.
    7. Save the 'data' and 'labels' lists into a single pickle file (DATASET_PATH).
    """
    data = []
    labels = []

    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please run data collection first.")
        return

    # Iterate over each class directory (e.g., '0', '1', '2'...)
    for dir_name in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_name)

        # Skip any files that might be in the root data dir (e.g., .DS_Store)
        if not os.path.isdir(dir_path):
            continue

        print(f"Processing class: {dir_name}")

        # Iterate over each image file (e.g., '0.jpg', '1.jpg'...)
        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)

            landmarks_per_image = []  # To store normalized (x, y) coords for this image
            x_coords = []  # To temporarily store raw x coords
            y_coords = []  # To temporarily store raw y coords

            # Read the image and convert it from BGR (OpenCV default) to RGB (MediaPipe)
            img = cv2.imread(img_full_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image with MediaPipe
            results = hands.process(img_rgb)

            # Check if any hands were detected
            if results.multi_hand_landmarks:
                # We only use the first hand detected (max_num_hands=1)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect all raw x, y coordinates
                    for landmark in hand_landmarks.landmark:
                        x_coords.append(landmark.x)
                        y_coords.append(landmark.y)

                # --- Normalization Step ---
                # We make all landmarks relative to the top-left-most point (min x, min y)
                # This makes the data independent of the hand's position in the frame.
                base_x, base_y = min(x_coords), min(y_coords)

                for x, y in zip(x_coords, y_coords):
                    # Append the relative x and y
                    landmarks_per_image.append(x - base_x)
                    landmarks_per_image.append(y - base_y)

                # Add the normalized landmark data for this one image to our main data list
                data.append(landmarks_per_image)
                # Add the corresponding class label (the directory name, e.g., '0')
                labels.append(dir_name)

    if not data:
        print("No data was processed. The dataset is empty. Did you run collection?")
        return

    print(f"Saving dataset to {DATASET_PATH}...")

    # Use 'with open' for safe file handling
    # 'wb' means 'write binary' mode, which is required for pickle
    with open(DATASET_PATH, "wb") as f:
        # Dump the data as a dictionary into the pickle file
        pickle.dump({"data": data, "labels": labels}, f)

    print("Dataset created successfully!")