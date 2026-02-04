import cv2
import mediapipe as mp
from typing import Optional, List

class MediaPipeHandler:
    """
    Interfaces with the MediaPipe library to detect hand positions in images.
    Provides standardized coordinates for further processing and model training.
    """
    def __init__(self, is_video_stream: bool = True):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=not is_video_stream,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5 if is_video_stream else 0.0
        )

    def process_frame(self, frame: cv2.typing.MatLike) -> Optional[List[float]]:
        """Identifies hand landmarks in a frame and returns their relative positions."""
        results = self.raw_process(frame)
        if not results or not results.multi_hand_landmarks:
            return None

        # Extract the raw coordinates of the first detected hand
        landmarks = results.multi_hand_landmarks[0]
        x_vals = [lm.x for lm in landmarks.landmark]
        y_vals = [lm.y for lm in landmarks.landmark]

        # Normalize positions relative to the hand's top-left corner
        origin_x, origin_y = min(x_vals), min(y_vals)
        normalized_points = []
        for x, y in zip(x_vals, y_vals):
            normalized_points.extend([x - origin_x, y - origin_y])

        return normalized_points

    def raw_process(self, frame: cv2.typing.MatLike):
        """Standard MediaPipe processing that returns the full results object."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.hands.process(rgb_frame)
        except Exception as e:
            print(f"Hand detection error: {e}")
            return None

    def draw_landmarks(self, frame: cv2.typing.MatLike, results):
        """Overlays the detected landmarks and skeletal connections onto the frame."""
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

    def close(self):
        """Releases the detection system resources."""
        self.hands.close()
