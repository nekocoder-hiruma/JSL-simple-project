import cv2
import numpy as np
from PIL import Image, ImageDraw

def draw_visual_feedback(frame: np.ndarray, text: str, position: tuple, font, color=(255, 255, 255)):
    """
    Overlays descriptive text onto the video frame to provide feedback to the user.
    Handles Japanese characters by bridging OpenCV with specialized drawing libraries.
    """
    if font:
        # Convert the frame format to allow for high-quality text rendering
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Draw the text using the provided font and color
        text_color = (color[2], color[1], color[0])  # Adjust color channels
        draw.text(position, text, font=font, fill=text_color)

        # Update the original frame with the new visual information
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        # Fallback to basic text display if specialized fonts are missing
        cv2.putText(frame, "Font Error", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
