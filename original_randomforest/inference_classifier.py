import pickle
from typing import Optional, Tuple, Any, List

import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw

from constants import (
    hands_video, mp_drawing, mp_drawing_styles, mp_hands,
    MODEL_PATH, LABELS_DICT
)

# --- Constants for drawing ---
BOX_WIDTH = 200
BOX_HEIGHT = 50
TEXT_PADDING = 10


def load_model(filepath: str) -> Optional[Any]:
    """
    Loads the trained model from a pickle file.

    Args:
        filepath (str): Path to the model.p file.

    Returns:
        Optional[Any]: The trained model, or None if the file is not found.
    """
    try:
        with open(filepath, "rb") as f:
            model_dict = pickle.load(f)
        return model_dict["model"]
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found. Please train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def initialize_webcam() -> Optional[cv2.VideoCapture]:
    """
    Initializes and opens the webcam.

    Returns:
        Optional[cv2.VideoCapture]: The VideoCapture object, or None if it fails.
    """
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None
    return cap


def find_font_path() -> Optional[str]:
    """
    Attempts to find a suitable font file for displaying Japanese
    characters on both Windows and macOS.
    """
    font_paths = [
        "C:/Windows/Fonts/YuGothM.ttc",  # Windows
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"  # macOS
    ]
    for path in font_paths:
        try:
            ImageFont.truetype(path, 32)
            print(f"Found font at: {path}")
            return path
        except IOError:
            continue
    print("Warning: No suitable Japanese font found. Falling back to default.")
    return None


def process_frame(frame: np.ndarray) -> Optional[Tuple[List[float], Any, Tuple[int, int]]]:
    """
    Processes a single video frame to find hand landmarks.

    Args:
        frame (np.ndarray): The video frame (BGR).

    Returns:
        Optional[Tuple[List[float], Any, Tuple[int, int]]]:
            - List[float]: Normalized landmark data (42 values).
            - Any: The hand_landmarks object from MediaPipe.
            - Tuple[int, int]: The frame height and width (H, W).
        Returns None if no hand is detected.
    """
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_video.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    # Draw landmarks on the frame (in-place)
    mp_drawing.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

    # --- Extract and Normalize Landmarks ---
    landmarks_data = []
    x_coords = []
    y_coords = []
    for landmark in hand_landmarks.landmark:
        x_coords.append(landmark.x)
        y_coords.append(landmark.y)

    base_x, base_y = min(x_coords), min(y_coords)
    for x, y in zip(x_coords, y_coords):
        landmarks_data.append(x - base_x)
        landmarks_data.append(y - base_y)

    return landmarks_data, hand_landmarks, (H, W)


def make_prediction(model: Any, landmarks_data: List[float]) -> str:
    """
    Makes a prediction using the trained model.

    Args:
        model (Any): The trained classifier.
        landmarks_data (List[float]): The normalized landmark data.

    Returns:
        str: The predicted Japanese character.
    """
    prediction = model.predict([np.asarray(landmarks_data)])
    predicted_character = LABELS_DICT[int(prediction[0])]
    return predicted_character


def draw_results(
        frame: np.ndarray,
        hand_landmarks: Any,
        predicted_char: str,
        font: ImageFont.FreeTypeFont,
        frame_shape: Tuple[int, int]
) -> None:
    """
    Draws the bounding boxes and prediction text on the frame (in-place).

    Args:
        frame (np.ndarray): The video frame to draw on.
        hand_landmarks (Any): The MediaPipe landmarks object.
        predicted_char (str): The character to display.
        font (ImageFont.FreeTypeFont): The PIL font object.
        frame_shape (Tuple[int, int]): The (Height, Width) of the frame.
    """
    H, W = frame_shape

    # --- Get Hand Bounding Box ---
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    x1 = int(min(x_coords) * W) - 10
    y1 = int(min(y_coords) * H) - 10
    x2 = int(max(x_coords) * W) + 10
    y2 = int(max(y_coords) * H) + 10

    # Draw the hand bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

    # --- Draw Fixed Prediction Text Box ---
    bg_tl = (x1, y1 - BOX_HEIGHT - 20)  # Top-left
    bg_br = (x1 + BOX_WIDTH, y1 - 20)  # Bottom-right

    if font:
        # Use PIL to draw Japanese text
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([bg_tl, bg_br], fill="white")
        text_y = bg_tl[1] + (BOX_HEIGHT - 32) // 2
        text_pos = (bg_tl[0] + TEXT_PADDING, text_y)
        draw.text(text_pos, predicted_char, font=font, fill=(0, 0, 0, 255))
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # Modify frame in-place

    else:
        # Fallback to OpenCV (no Japanese support)
        text = str(predicted_char)  # Will just show the char
        cv2.rectangle(frame, bg_tl, bg_br, (255, 255, 255), cv2.FILLED)
        text_y = bg_tl[1] + (BOX_HEIGHT + 25) // 2  # Approx center
        text_origin = (bg_tl[0] + TEXT_PADDING, text_y)
        cv2.putText(frame, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX,
                    1.3, (0, 0, 0), 3, cv2.LINE_AA)


def run_inference() -> None:
    """
    Main orchestrator function for the inference pipeline.
    Initializes, runs the main loop, and cleans up.
    """
    model = load_model(MODEL_PATH)
    cap = initialize_webcam()
    font_path = find_font_path()
    font = ImageFont.truetype(font_path, 32) if font_path else None

    if model is None or cap is None:
        print("Exiting due to initialization error.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # Process frame to get landmarks
        processed_data = process_frame(frame)

        if processed_data:
            landmarks_data, hand_landmarks, frame_shape = processed_data

            # Make prediction based on landmarks
            predicted_char = make_prediction(model, landmarks_data)

            # Draw all results on the frame
            draw_results(frame, hand_landmarks, predicted_char, font, frame_shape)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()