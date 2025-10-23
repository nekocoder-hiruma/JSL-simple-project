from typing import Optional, List

import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw

from config_loader import hands


def find_font_path() -> Optional[str]:
    """
    Attempts to find a suitable font file on Windows or macOS.
    This is crucial for displaying non-ASCII characters like Japanese.
    """
    font_paths = [
        "C:/Windows/Fonts/YuGothM.ttc",  # Windows (Yu Gothic Medium)
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"  # macOS (Hiragino Kaku Gothic W3)
    ]
    for path in font_paths:
        try:
            ImageFont.truetype(path, 32)
            print(f"Found font at: {path}")
            return path
        except IOError:
            continue
    print("Warning: No suitable Japanese font found. Text may not display correctly.")
    return None


def draw_text(frame: np.ndarray, text: str, position: tuple,
              font: ImageFont.FreeTypeFont, color: tuple = (255, 255, 255)) -> None:
    """
    Draws text (including Japanese characters) on an OpenCV frame using PIL.

    Args:
        frame (np.ndarray): The OpenCV frame to draw on (modified in-place).
        text (str): The text to draw.
        position (tuple): The (x, y) top-left position for the text.
        font (ImageFont.FreeTypeFont): The PIL font object.
        color (tuple): The BGR color for the text.
    """
    if font:
        # Convert OpenCV frame (BGR) to PIL Image (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Draw the text using PIL
        # PIL uses RGB, so we convert the BGR color
        pil_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=pil_color)

        # Convert back to OpenCV frame and update the original frame
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        # Fallback to OpenCV's putText if no font is found (will show '???')
        cv2.putText(frame, "Font not found", position, cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_AA)


def process_frame(frame: cv2.typing.MatLike) -> Optional[List[float]]:
    """
    Processes a single frame with MediaPipe and returns normalized hand landmarks.
    This is used during both data collection and inference.
    """
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Step 1: Extract all landmark coordinates into separate lists.
        x_coords = []
        y_coords = []
        for landmark in hand_landmarks.landmark:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)

        # Step 2: Find the minimum coordinate to use as the origin for normalization.
        # This makes the gesture data independent of its position on the screen.
        base_x = min(x_coords)
        base_y = min(y_coords)

        # Step 3: Normalize and flatten the coordinates into a single list.
        normalized_landmarks = []
        for x, y in zip(x_coords, y_coords):
            # Calculate the position relative to the hand's top-left corner.
            normalized_x = x - base_x
            normalized_y = y - base_y

            # Append the normalized coordinates to the final list.
            normalized_landmarks.append(normalized_x)
            normalized_landmarks.append(normalized_y)

        return normalized_landmarks

    except Exception as e:
        print(f"Error processing frame: {e}")
        return None
