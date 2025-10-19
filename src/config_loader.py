import configparser
import csv
import os
from typing import List

import mediapipe as mp

# --- Load Configuration ---
config = configparser.ConfigParser()
# Override base config with second config in a list when needed
config.read(['configs/base_config.ini', 'configs/keras_config.ini'])

# --- [paths] ---
DATA_DIR = config.get('paths', 'data_dir')
DATASET_PATH = config.get('paths', 'dataset_path')
RF_MODEL_PATH = config.get('paths', 'rf_model_path')
MODEL_PATH = config.get('paths', 'keras_model_path')
LABELS_PATH = config.get('paths', 'labels_path')

# --- [camera] ---
CAMERA_INDEX = config.getint('camera', 'index')

# --- [model_params] ---
NUMBER_OF_CLASSES = config.getint('model_params', 'number_of_classes')
SEQUENCE_LENGTH = config.getint('model_params', 'sequence_length')

# --- [data_collection] ---
DATASET_SIZE_PER_CLASS = config.getint('data_collection', 'samples_per_class')
CLASSES_TO_COLLECT_STR = config.get('data_collection', 'classes_to_collect', fallback='').strip()

# --- [inference] ---
CONFIDENCE_THRESHOLD = config.getfloat('inference', 'confidence_threshold')
HIGH_CONFIDENCE_THRESHOLD = config.getfloat('inference', 'high_confidence_threshold')


def parse_classes_to_collect(class_str: str, total_classes: int) -> List[int]:
    """
    Parses 'classes_to_collect' from base_config.ini with enhanced logic.
    - Empty string: Collect all classes from 0.
    - Single integer (e.g., '15'): Collect all classes from that index onwards.
    - Comma-separated list (e.g., '0, 5, 15'): Collect only those specific classes.
    """
    # Case 1: Empty string -> Collect all
    if not class_str:
        print("Config 'classes_to_collect' is empty. Collecting all classes.")
        return list(range(total_classes))

    # Case 2: Comma-separated list -> Collect specific classes
    if ',' in class_str:
        try:
            class_indices = [int(x.strip()) for x in class_str.split(',')]
            valid_indices = [idx for idx in class_indices if 0 <= idx < total_classes]

            if len(valid_indices) != len(class_indices):
                print("Warning: Some class indices were out of range and have been skipped.")

            print(f"Collecting specific classes: {sorted(set(valid_indices))}")
            return sorted(set(valid_indices))
        except ValueError:
            print("Error: Invalid list format for 'classes_to_collect'. Defaulting to all classes.")
            return list(range(total_classes))

    # Case 3: Single integer -> Collect from that index onwards
    else:
        try:
            start_index = int(class_str)
            if 0 <= start_index < total_classes:
                print(f"Collecting all classes starting from index {start_index}.")
                return list(range(start_index, total_classes))
            else:
                print(
                    f"Warning: Start index {start_index} is out of range "
                    f"(0-{total_classes - 1}). Defaulting to all classes.")
                return list(range(total_classes))
        except ValueError:
            print("Error: Invalid single integer format for 'classes_to_collect'. "
                  "Defaulting to all classes.")
            return list(range(total_classes))


# --- Dynamically parse the list of classes to collect ---
CLASSES_TO_COLLECT = parse_classes_to_collect(CLASSES_TO_COLLECT_STR, NUMBER_OF_CLASSES)


def load_labels_from_csv(filepath: str) -> dict:
    """
    Loads class labels from a CSV file into a dictionary.
    The CSV should have two columns with a header: 'index' and 'label'.
    """
    if not os.path.exists(filepath):
        print(f"Error: Labels file not found at '{filepath}'. Using an empty dictionary.")
        return {}

    labels_dict = {}
    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip the header row
            for row in reader:
                if row:  # Ensure the row is not empty
                    labels_dict[int(row[0])] = row[1]
    except Exception as e:
        print(f"Error reading labels from {filepath}: {e}. Please check the file format.")
        return {}

    print(f"Successfully loaded {len(labels_dict)} labels from {filepath}.")
    return labels_dict


# --- Dynamically Loaded Labels ---
# The LABELS_DICT is now created by reading the CSV file specified in base_config.ini
LABELS_DICT = load_labels_from_csv(LABELS_PATH)

"""
--- Static Project Constants (Not in config file) ---
"""

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


"""
Initialize the MediaPipe Hands solution.
'hands' is used for processing static images (dataset creation).
- static_image_mode=True: Optimizes for individual images (not a video stream).
- max_num_hands=1: We are only tracking one hand for these signs.
- min_detection_confidence=0.5: The minimum confidence (50%) for a hand
  detection to be considered successful.
"""
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

"""
'hands_video' is used for processing the real-time video stream (inference).
- static_image_mode=False: Optimizes for a video stream.
- min_tracking_confidence=0.5: The minimum confidence (50%) for the hand
  landmarks to be successfully tracked from one frame to the next.
"""
hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)

