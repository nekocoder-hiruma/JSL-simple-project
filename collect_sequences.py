import os
import pickle
from typing import Optional, List, Dict, Any

import cv2

from constants import (DATA_DIR, NUMBER_OF_CLASSES, DATASET_SIZE_PER_CLASS,
                       LABELS_DICT, SEQUENCE_LENGTH)
from helpers import process_frame


def wait_for_s_key(cap: cv2.VideoCapture, class_index: int) -> None:
    """
    Displays a prompt on the webcam feed and waits for the user to press 's'.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        prompt = f"Press 'S' to start recording for '{LABELS_DICT[class_index]}'"
        cv2.putText(frame, prompt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)

        if cv2.waitKey(25) == ord('s'):
            break


def record_sequence(cap: cv2.VideoCapture) -> Optional[List[List[float]]]:
    """
    Records a single sequence of SEQUENCE_LENGTH frames, processing each one.
    """
    sequence_of_landmarks = []
    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame during sequence recording.")
            return None

        # Display "Recording..." feedback to the user
        cv2.putText(frame, "RECORDING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)  # Necessary to update the window

        landmarks, _ = process_frame(frame)
        if landmarks:
            sequence_of_landmarks.append(landmarks)
        else:
            # If a hand is lost mid-sequence, append a zero vector
            # This is important for the LSTM later
            sequence_of_landmarks.append([0.0] * 42)

    return sequence_of_landmarks


def initialize_resources() -> Optional[Dict[str, Any]]:
    """
    Initializes the data directory and webcam.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None

    return {"cap": cap, "data": [], "labels": []}


def run_collection_loop(resources: Dict[str, Any]) -> None:
    """
    Runs the main nested loops to collect data for all classes and samples.
    """
    cap = resources["cap"]
    for class_idx in range(NUMBER_OF_CLASSES):
        print(f"\n--- Preparing to collect data for class {class_idx} ('{LABELS_DICT[class_idx]}') ---")

        for sample_num in range(DATASET_SIZE_PER_CLASS):
            wait_for_s_key(cap, class_idx)

            print(f"Recording sample {sample_num + 1}/{DATASET_SIZE_PER_CLASS} "
                  f"for class '{LABELS_DICT[class_idx]}'")

            sequence = record_sequence(cap)

            if sequence:
                resources["data"].append(sequence)
                resources["labels"].append(class_idx)


def save_data(resources: Dict[str, Any]) -> None:
    """
    Saves the collected data and labels to a pickle file.
    """
    print("\nData collection complete. Saving to data.pickle...")
    try:
        with open("data.pickle", "wb") as f:
            pickle.dump({"data": resources["data"], "labels": resources["labels"]}, f)
        print("Dataset saved successfully!")
    except IOError as e:
        print(f"Error saving dataset: {e}")


def cleanup_resources(resources: Dict[str, Any]) -> None:
    """
    Releases the webcam and destroys all OpenCV windows.
    """
    print("Cleaning up resources...")
    resources["cap"].release()
    cv2.destroyAllWindows()


def collect_sequences() -> None:
    """
    Main orchestrator for collecting and processing sequences of landmark data.
    This function's sole responsibility is to call other functions in the correct order.
    """
    resources = initialize_resources()
    if not resources:
        return

    try:
        run_collection_loop(resources)
        save_data(resources)
    except Exception as e:
        print(f"An unexpected error occurred during collection: {e}")
    finally:
        # Ensure cleanup happens even if an error occurs
        cleanup_resources(resources)