import os
import cv2
import numpy as np
from core.config_loader import ConfigLoader
from core.resource_manager import ResourceManager
from core.mediapipe_utils import MediaPipeHandler
from utils.helpers import draw_visual_feedback

class BaseCollector:
    """
    Common logic for capturing hand movement data via webcam.
    Handles setup and cleanup for different types of sign language data.
    """
    def __init__(self, config: ConfigLoader, resource_manager: ResourceManager):
        self.config = config
        self.rm = resource_manager
        self.data_dir = config.get_setting('paths', 'data_dir')
        self.num_classes = config.get_int('model_params', 'number_of_classes', 0)
        self.samples_per_class = config.get_int('data_collection', 'samples_per_class', 0)

    def prepare_directory(self):
        """Ensures the storage location for collected data exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def get_classes_to_collect(self) -> list[int]:
        """Parses the configuration to determine which labels the user wants to gather."""
        raw_classes = self.config.get_setting('data_collection', 'classes_to_collect', '')
        if not raw_classes.strip():
            return list(range(self.num_classes))
        
        try:
            if ',' in raw_classes:
                return [int(c.strip()) for c in raw_classes.split(',')]
            else:
                start_from = int(raw_classes.strip())
                return list(range(start_from, self.num_classes))
        except ValueError:
            print("Warning: Invalid classes_to_collect format. Defaulting to all classes.")
            return list(range(self.num_classes))

    def draw_interactive_overlay(self, frame, text, instructions=None, progress=None):
        """Draws status labels, progress bars, and hotkey hints onto the frame."""
        font = self.rm.get_japanese_font()
        
        # Draw main status text (e.g., Character Name)
        draw_visual_feedback(frame, text, (20, 40), font, color=(0, 255, 0))
        
        # Draw dynamic instructions hint if provided
        if instructions:
            cv2.putText(frame, instructions, (20, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw a visual progress bar if progress data exists (current, total)
        if progress:
            curr, total = progress
            bar_width = 200
            bar_height = 20
            start_x, start_y = 20, 70
            
            # Draw background/border
            cv2.rectangle(frame, (start_x, start_y), (start_x + bar_width, start_y + bar_height), (50, 50, 50), -1)
            # Draw fill based on percentage
            fill = int((curr / total) * bar_width)
            cv2.rectangle(frame, (start_x, start_y), (start_x + fill, start_y + bar_height), (0, 200, 0), -1)
            cv2.putText(frame, f"{curr}/{total}", (start_x + bar_width + 10, start_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def wait_for_start(self, cap, prompt: str):
        """Pauses the process until the user is ready to begin capturing."""
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            self.draw_interactive_overlay(frame, prompt, instructions="Press 'E' to Entry / Start | 'Q' to Quit")
            cv2.imshow("Capture Window", frame)
            
            key = cv2.waitKey(25) & 0xFF
            if key == ord('e'):
                return True
            if key == ord('q'):
                return False
        return False
