import os
import pickle
import cv2
from core.mediapipe_utils import MediaPipeHandler

class DatasetCreator:
    """
    Converts raw image files into a structured dataset for model training.
    Scans directories, extracts hand landmarks, and saves the binary data.
    """
    def __init__(self, config, mp_handler: MediaPipeHandler):
        self.config = config
        self.mp_handler = mp_handler
        self.data_dir = config.get_setting('paths', 'data_dir')
        self.save_path = config.get_setting('paths', 'dataset_path')

    def create(self):
        """Processes all images in the source directory and saves the final dataset."""
        dataset, labels = [], []

        if not os.path.exists(self.data_dir):
            print("Source directory not found.")
            return

        for folder_name in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            print(f"Processing category: {folder_name}")
            for file_name in os.listdir(folder_path):
                if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img = cv2.imread(os.path.join(folder_path, file_name))
                if img is None: continue

                points = self.mp_handler.process_frame(img)
                if points:
                    dataset.append(points)
                    labels.append(folder_name)

        if dataset:
            with open(self.save_path, "wb") as f:
                pickle.dump({"data": dataset, "labels": labels}, f)
            print(f"Dataset successfully created at {self.save_path}")
        else:
            print("No valid data found to create a dataset.")
