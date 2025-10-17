from typing import Optional, List, Tuple

import cv2

from constants import hands


def process_frame(frame: cv2.typing.MatLike) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Processes a single frame with MediaPipe and returns normalized hand landmarks.
    """
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            base_x, base_y = min(x_coords), min(y_coords)

            # Create a clear, flat list for the normalized coordinates
            normalized_landmarks = []
            for x, y in zip(x_coords, y_coords):
                normalized_landmarks.append(x - base_x)
                normalized_landmarks.append(y - base_y)

            return normalized_landmarks, hand_landmarks
        else:
            return None, None
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None, None