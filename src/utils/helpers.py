import cv2
import numpy as np
from PIL import Image, ImageDraw

def draw_visual_feedback(frame: np.ndarray, text: str, position: tuple, font, color=(255, 255, 255)):
    """
    Overlays descriptive text onto the video frame with a background block for legibility.
    Handles Japanese characters by bridging OpenCV with specialized drawing libraries.
    """
    if font:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Calculate text area for the background block
        bbox = draw.textbbox(position, text, font=font)
        # Add slight padding to the block
        padding = 5
        rect_coords = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
        
        # Draw solid background block for contrast
        draw.rectangle(rect_coords, fill=(0, 0, 0))
        
        # Draw the text on top
        text_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=text_color)

        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    else:
        # Fallback to standard CV2 text if Japanese fonts are unavailable
        cv2.rectangle(frame, (position[0]-5, position[1]-30), (position[0]+300, position[1]+10), (0,0,0), -1)
        cv2.putText(frame, f"Font Error: {text}", position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
