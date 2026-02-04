import os
import cv2
from core.config_loader import ConfigLoader
from core.resource_manager import ResourceManager
from core.mediapipe_utils import MediaPipeHandler

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

    def wait_for_start(self, cap, prompt: str):
        """Pauses the process until the user is ready to begin capturing."""
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            cv2.putText(frame, prompt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Capture Window", frame)
            
            if cv2.waitKey(25) == ord('s'):
                return True
            if cv2.waitKey(25) == ord('q'):
                return False
        return False
