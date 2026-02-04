import os
import cv2
from typing import Optional
from PIL import ImageFont

class ResourceManager:
    """
    Manages shared resources like the webcam and specialized fonts.
    This ensures that resources are opened and closed properly without duplication.
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.japanese_font = None

    def get_webcam(self) -> cv2.VideoCapture:
        """Opens and returns the webcam stream."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap

    def get_japanese_font(self, size: int = 32) -> Optional[ImageFont.FreeTypeFont]:
        """Loads a font capable of displaying Japanese characters."""
        if self.japanese_font:
            return self.japanese_font

        font_paths = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # macOS
            "C:/Windows/Fonts/YuGothM.ttc"  # Windows
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    self.japanese_font = ImageFont.truetype(path, size)
                    return self.japanese_font
                except IOError:
                    continue
        return None

    def release_all(self):
        """Releases the webcam and closes any active windows."""
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
