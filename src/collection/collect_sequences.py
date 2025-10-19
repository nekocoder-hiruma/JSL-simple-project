import os
import pickle
from typing import Optional, List, Dict, Any

import cv2
from PIL import ImageFont

from config_loader import (
    DATA_DIR, DATASET_SIZE_PER_CLASS,
    LABELS_DICT, SEQUENCE_LENGTH, CAMERA_INDEX, CLASSES_TO_COLLECT, DATASET_PATH
)
from utils.helpers import find_font_path, draw_text, process_frame


def draw_progress_bar(frame: cv2.typing.MatLike, progress: float) -> None:
    """
    Draws a progress bar on the bottom of the frame.

    :param frame:
    :param progress:
    :return:
    """
    h, w, _ = frame.shape
    bar_y = h - 30
    bar_w = int(w * 0.8)
    bar_x = int((w - bar_w) / 2)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (100, 100, 100), -1)
    progress_w = int(bar_w * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_w, bar_y + 20), (0, 255, 0), -1)


def record_sequence(cap: cv2.VideoCapture) -> Optional[List[List[float]]]:
    """
    Records a single sequence with a visual progress bar.

    :param cap: cv2 VideoCapture object
    :return: Optional List of landmarks coordinates
    """
    sequence_of_landmarks = []
    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret: return None

        progress = (frame_num + 1) / SEQUENCE_LENGTH
        draw_progress_bar(frame, progress)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

        landmarks = process_frame(frame)
        sequence_of_landmarks.append(landmarks or ([0.0] * 42))

    return sequence_of_landmarks


def review_and_confirm(cap: cv2.VideoCapture, class_index: int,
                       font: ImageFont.FreeTypeFont) -> bool:
    """
    Shows a review screen and waits for user confirmation.

    :param cap: cv2 VideoCapture object
    :param class_index: Class index to display the character that code has collected
    :param font: Font typing for Japanese display
    :return: boolean indicating true or false
    """
    start_time = cv2.getTickCount()
    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < 2.5:
        ret, frame = cap.read()
        if not ret: break

        prompt1 = f"Collected '{LABELS_DICT.get(class_index, 'Unknown')}' Press 'D' to discard."
        draw_text(frame, prompt1, (50, 50), font, (0, 255, 255))
        cv2.imshow("frame", frame)

        key = cv2.waitKey(50)
        if key == ord('d'):
            print("Sample discarded.")
            return False
    return True


def inter_sample_pause(cap: cv2.VideoCapture, font: ImageFont.FreeTypeFont) -> None:
    """
    Displays a countdown timer between samples to allow the user to reset.

    :param cap: cv2 VideoCapture object
    :param font: Font Typing for Japanese display
    :return:
    """
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret: break
        prompt = f"Next sample in {i}..."
        draw_text(frame, prompt, (50, 50), font, (255, 255, 255))
        cv2.imshow("frame", frame)
        cv2.waitKey(1000)


def wait_for_class_start(cap: cv2.VideoCapture, class_index: int,
                         font: ImageFont.FreeTypeFont) -> bool:
    """
    Displays a prompt to start collecting for a new class.

    :param cap: cv2 VideoCapture object
    :param class_index: Class index to display the character that code is collecting
    :param font: Font typing for Japanese display
    :return: Boolean to indicate if the code continues or ends
    """
    while True:
        ret, frame = cap.read()
        if not ret: break

        prompt1 = f"Press 'S' to collect for '{LABELS_DICT.get(class_index, 'Unknown')}'"
        prompt2 = "Press 'Q' to quit and save"
        draw_text(frame, prompt1, (50, 50), font, (0, 255, 0))
        draw_text(frame, prompt2, (50, 100), font, (0, 255, 255))
        cv2.imshow("frame", frame)

        key = cv2.waitKey(25)
        if key == ord('s'):
            return True
        if key == ord('q'):
            break
    return False


def initialize_resources() -> Optional[Dict[str, Any]]:
    """Initializes resources and loads existing data if found."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open video stream for camera index {CAMERA_INDEX}.")
        return None

    font_path = find_font_path()
    font = ImageFont.truetype(font_path, 32) if font_path else None

    data, labels = [], []
    if os.path.exists(DATASET_PATH):
        print(f"Existing dataset found at '{DATASET_PATH}'. Loading data...")
        with open(DATASET_PATH, "rb") as f:
            dataset = pickle.load(f)
        data, labels = dataset["data"], dataset["labels"]
    else:
        print("No existing dataset found. Starting a new one.")

    return {"cap": cap, "data": data, "labels": labels, "font": font}


def handle_sample_collection(resources: Dict[str, Any], class_idx: int) -> bool:
    """
    Orchestrates the process of collecting a single sample: record, review, and append.
    Returns True if a sample was successfully collected and confirmed, False otherwise.

    :param resources: Resources dict to retrieve cv2 VideoCapture and Font
    :param class_idx: Class index to get the character for the collected data
    :return: Boolean
    """
    sequence = record_sequence(resources["cap"])

    if sequence and review_and_confirm(resources["cap"], class_idx, resources["font"]):
        resources["data"].append(sequence)
        resources["labels"].append(class_idx)
        return True
    return False


def run_collection_for_class(resources: Dict[str, Any], class_idx: int) -> None:
    """
    Runs the data collection loop for a single class.

    :param resources: Resources dict to retrieve cv2 VideoCapture and Font
    :param class_idx: Class index to display the character
    :return:
    """
    label = LABELS_DICT.get(class_idx, None)
    if not label:
        return
    print(f"\n--- Preparing to collect data for class {class_idx} ('{label}') ---")

    sample_count = 0
    while sample_count < DATASET_SIZE_PER_CLASS:
        if sample_count > 0:
            inter_sample_pause(resources["cap"], resources["font"])

        print(f"Recording sample {sample_count + 1}/{DATASET_SIZE_PER_CLASS} for class '{label}'")
        if handle_sample_collection(resources, class_idx):
            sample_count += 1
            print(f"Sample {sample_count}/{DATASET_SIZE_PER_CLASS} for '{label}' saved.")


def save_data(resources: Dict[str, Any]) -> None:
    """Saves the collected data and labels to a pickle file."""
    if not resources["data"]:
        print("No data collected. Exiting without saving.")
        return

    print(f"\nSaving all data to '{DATASET_PATH}'...")
    try:
        with open(DATASET_PATH, "wb") as f:
            pickle.dump({"data": resources["data"], "labels": resources["labels"]}, f)
        print("Dataset saved successfully!")
    except IOError as e:
        print(f"Error saving dataset: {e}")


def cleanup_resources(resources: Dict[str, Any]) -> None:
    """Releases the webcam and destroys all OpenCV windows."""
    print("Cleaning up resources...")
    resources["cap"].release()
    cv2.destroyAllWindows()


def collect_sequences() -> None:
    """Main orchestrator for the data collection pipeline."""
    resources = initialize_resources()
    if not resources: return

    try:
        for class_idx in CLASSES_TO_COLLECT:
            if not wait_for_class_start(resources["cap"], class_idx, resources["font"]):
                print("Quit signal received. Proceeding to save.")
                break
            run_collection_for_class(resources, class_idx)

        save_data(resources)
    except Exception as e:
        print(f"An unexpected error occurred during collection: {e}")
    finally:
        cleanup_resources(resources)