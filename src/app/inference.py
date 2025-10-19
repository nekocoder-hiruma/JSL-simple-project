from collections import deque
from typing import Optional, List, Any, Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFont

from config_loader import (
    MODEL_PATH,
    LABELS_DICT,
    SEQUENCE_LENGTH,
    hands_video,  # Note: hands_video is now used instead of hands
    mp_drawing,
    mp_hands,
    mp_drawing_styles, CAMERA_INDEX, CONFIDENCE_THRESHOLD, HIGH_CONFIDENCE_THRESHOLD
)
# Import shared functions from the new helpers file
from utils.helpers import find_font_path, draw_text

# --- Constants for drawing ---
BOX_WIDTH = 200
BOX_HEIGHT = 50
TEXT_PADDING = 10


def load_keras_model(filepath: str) -> Optional[tf.keras.Model]:
    """Loads the trained Keras model from an .h5 file."""
    try:
        print(f"Loading model from {filepath}...")
        model = tf.keras.models.load_model(filepath)
        print("Model loaded successfully.")
        return model
    except (FileNotFoundError, IOError):
        print(f"Error: Model file not found at '{filepath}'. Please train the model first.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return None


def initialize_resources() -> Optional[Dict[str, Any]]:
    """Initializes webcam, font, and the sequence buffer."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None

    font_path = find_font_path()
    font = ImageFont.truetype(font_path, 32) if font_path else None

    return {
        "cap": cap,
        "font": font,
        "sequence_buffer": deque(maxlen=SEQUENCE_LENGTH)
    }


def process_and_draw_landmarks(frame: np.ndarray) -> Tuple[Optional[List[float]], Optional[Any]]:
    """
    Processes a frame to get landmarks and also draws them on the frame.
    This is different from helpers.process_frame, which only returns data.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_video.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(
        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

    # Normalize landmarks for prediction
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    base_x, base_y = min(x_coords), min(y_coords)

    normalized_landmarks = [coord for pos in zip(x_coords, y_coords) for coord in (pos[0] - base_x, pos[1] - base_y)]
    return normalized_landmarks, hand_landmarks


def make_sequence_prediction(model: tf.keras.Model, sequence: np.ndarray) -> Tuple[Optional[str], float]:
    """Makes a prediction on a full sequence of landmark data."""
    input_data = np.expand_dims(sequence, axis=0)
    prediction = model.predict(input_data, verbose=0)[0]

    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    if confidence > CONFIDENCE_THRESHOLD:
        return LABELS_DICT[class_index], confidence

    return None, confidence


def draw_prediction_on_frame(frame: np.ndarray, hand_landmarks: Any, predicted_char: str,
                             font: ImageFont.FreeTypeFont) -> None:
    """Draws the bounding box and the predicted character on the frame."""
    H, W, _ = frame.shape
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    x1, y1 = int(min(x_coords) * W) - 10, int(min(y_coords) * H) - 10

    bg_tl = (x1, y1 - BOX_HEIGHT - 20)
    bg_br = (x1 + BOX_WIDTH, y1 - 20)

    # Draw the white background for the text
    cv2.rectangle(frame, bg_tl, bg_br, (255, 255, 255), cv2.FILLED)

    # Use the shared helper function to draw the text
    text_y = bg_tl[1] + (BOX_HEIGHT - 32) // 2
    text_pos = (bg_tl[0] + TEXT_PADDING, text_y)
    draw_text(frame, predicted_char, text_pos, font, (0, 0, 0))  # Black text


def run_inference() -> None:
    """Main orchestrator for the real-time inference pipeline."""
    model = load_keras_model(MODEL_PATH)
    resources = initialize_resources()

    if not model or not resources:
        print("Exiting due to initialization failure.")
        return

    cap, font, sequence_buffer = resources["cap"], resources["font"], resources["sequence_buffer"]
    latest_prediction = ""
    prediction_counter = 0
    prediction_interval = 1

    while True:
        ret, frame = cap.read()
        if not ret: break

        landmarks, hand_landmarks_obj = process_and_draw_landmarks(frame)
        sequence_buffer.append(landmarks or ([0.0] * 42))

        if len(sequence_buffer) == SEQUENCE_LENGTH and prediction_counter == 0:
            predicted_char, confidence = make_sequence_prediction(model, np.array(sequence_buffer))

            if predicted_char:
                latest_prediction = predicted_char
                prediction_interval = 10 if confidence > HIGH_CONFIDENCE_THRESHOLD else 1
            else:
                prediction_interval = 1

        prediction_counter = (prediction_counter + 1) % prediction_interval

        if latest_prediction and hand_landmarks_obj:
            draw_prediction_on_frame(frame, hand_landmarks_obj, latest_prediction, font)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()