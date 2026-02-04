import pickle
from typing import List, Optional
import cv2
from data.collection.base_collector import BaseCollector

class DynamicCollector(BaseCollector):
    """
    Records sequences of hand movements to represent dynamic signs.
    Extracts hand positions across multiple frames and saves the movement data.
    """
    def __init__(self, config, resource_manager, mp_handler):
        super().__init__(config, resource_manager)
        self.mp_handler = mp_handler
        self.seq_len = config.get_int('model_params', 'sequence_length', 30)
        self.save_path = config.get_setting('paths', 'dataset_path')

    def record_movement(self, cap) -> Optional[List[List[float]]]:
        """Captures a single motion sequence and converts it into numerical data."""
        movement_data = []
        for _ in range(self.seq_len):
            ret, frame = cap.read()
            if not ret: return None

            cv2.imshow("Capture Window", frame)
            cv2.waitKey(1)

            positions = self.mp_handler.process_frame(frame)
            movement_data.append(positions or ([0.0] * 42))
        return movement_data

    def collect(self):
        """Orchestrates the recording of multiple movement samples for each sign."""
        self.prepare_directory()
        cap = self.rm.get_webcam()
        all_data, all_labels = [], []

        for class_idx in range(self.num_classes):
            prompt = f"Motion for Label {class_idx}: Press 'S' to Start"
            if not self.wait_for_start(cap, prompt):
                break

            for sample_idx in range(self.samples_per_class):
                print(f"Recording Sample {sample_idx + 1}...")
                sequence = self.record_movement(cap)
                if sequence:
                    all_data.append(sequence)
                    all_labels.append(class_idx)

        # Save the finalized movement dataset
        with open(self.save_path, "wb") as f:
            pickle.dump({"data": all_data, "labels": all_labels}, f)
        
        self.rm.release_all()
