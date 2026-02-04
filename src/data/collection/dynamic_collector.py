import os
import cv2
import pickle
import csv
from typing import List, Optional
from data.collection.base_collector import BaseCollector

class DynamicCollector(BaseCollector):
    """
    Records sequences of hand movements to represent dynamic signs.
    Implements a flow where users capture multiple sequences per sign and combine them with static data.
    """
    def __init__(self, config, resource_manager, mp_handler):
        super().__init__(config, resource_manager)
        self.mp_handler = mp_handler
        self.seq_len = config.get_int('model_params', 'sequence_length', 30)
        self.save_path = config.get_setting('paths', 'dataset_path')

    def load_labels(self):
        """Creates a mapping from class index to the actual sign character."""
        labels_dict = {}
        labels_path = self.config.get_setting('paths', 'labels_path')
        try:
            with open(labels_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if row: labels_dict[int(row[0])] = row[1]
        except Exception as e:
            print(f"Error loading labels: {e}")
        return labels_dict

    def record_movement(self, cap, sign_text, batch_progress) -> Optional[List[List[float]]]:
        """Captures a 30-frame motion sequence with visual recording progress."""
        movement_data = []
        for i in range(self.seq_len):
            ret, frame = cap.read()
            if not ret: return None

            # Draw skeletal overlay
            results = self.mp_handler.raw_process(frame)
            self.mp_handler.draw_landmarks(frame, results)

            # Draw "Recording..." status and a mini recording progress bar
            self.draw_interactive_overlay(frame, f"Recording: {sign_text}", 
                                       instructions="Keep moving your hand...",
                                       progress=batch_progress)
            
            # Draw frame-level recording progress (percent of 30 frames)
            cv2.rectangle(frame, (20, 100), (220, 110), (0, 0, 100), -1)
            fill = int(((i + 1) / self.seq_len) * 200)
            cv2.rectangle(frame, (20, 100), (20 + fill, 110), (0, 0, 255), -1)

            cv2.imshow("Capture Window", frame)
            cv2.waitKey(1)

            positions = self.mp_handler.process_frame(frame)
            movement_data.append(positions or ([0.0] * 42))
        return movement_data

    def merge_static_data(self, dynamic_data, dynamic_labels):
        """Integrates previously captured static landmark sequences into the dynamic dataset."""
        # Find static data in the same directory as the dynamic dataset
        dataset_dir = os.path.dirname(self.save_path)
        static_path = os.path.join(dataset_dir, "static_jsl_data.pickle")
        if os.path.exists(static_path):
            print(f"Merging static sequence data from {static_path}...")
            with open(static_path, "rb") as f:
                static_content = pickle.load(f)
            
            # Both are now sequences of shape (seq_len, 42), so we append directly
            dynamic_data.extend(static_content.get("data", []))
            dynamic_labels.extend(static_content.get("labels", []))
        return dynamic_data, dynamic_labels

    def collect(self):
        """Interactive loop to collect motion sequences sample by sample."""
        self.prepare_directory()
        cap = self.rm.get_webcam()
        labels_dict = self.load_labels()
        classes = self.get_classes_to_collect()
        
        new_data, new_labels = [], []

        if not self.wait_for_start(cap, "Dynamic Collection Ready"):
            self.rm.release_all()
            return

        for class_idx in classes:
            char_name = labels_dict.get(class_idx, str(class_idx))
            current_character_samples = []
            
            while True:
                ret, frame = cap.read()
                if not ret: break

                results = self.mp_handler.raw_process(frame)
                self.mp_handler.draw_landmarks(frame, results)

                count = len(current_character_samples)
                batch_progress = (count, self.samples_per_class)
                
                status = f"Collecting: {char_name}"
                if count < self.samples_per_class:
                    instr = "Press 'C' to Start Recording Sequence | 'Q' to Exit"
                else:
                    status = f"Done with: {char_name}"
                    instr = "Press 'S' to Save & Next Character | 'Q' to Exit"

                self.draw_interactive_overlay(frame, status, instructions=instr, progress=batch_progress)
                cv2.imshow("Capture Window", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and count < self.samples_per_class:
                    sequence = self.record_movement(cap, char_name, batch_progress)
                    if sequence:
                        current_character_samples.append(sequence)
                
                if key == ord('s') and count >= self.samples_per_class:
                    # Finalize this character and move to next
                    new_data.extend(current_character_samples)
                    new_labels.extend([class_idx] * len(current_character_samples))
                    break
                
                if key == ord('q'):
                    # Save what we have and quit
                    new_data.extend(current_character_samples)
                    new_labels.extend([class_idx] * len(current_character_samples))
                    break
            
            if key == ord('q'): break

        # Incremental saving logic
        if new_data:
            all_data, all_labels = [], []
            if os.path.exists(self.save_path):
                print(f"Loading existing dynamic dataset from {self.save_path}...")
                with open(self.save_path, "rb") as f:
                    old_dataset = pickle.load(f)
                    all_data = old_dataset.get('data', [])
                    all_labels = old_dataset.get('labels', [])
            
            all_data.extend(new_data)
            all_labels.extend(new_labels)
            
            # Final merge with static sequences
            all_data, all_labels = self.merge_static_data(all_data, all_labels)
            
            with open(self.save_path, "wb") as f:
                pickle.dump({"data": all_data, "labels": all_labels}, f)
            print(f"Dynamic dataset updated successfully at {self.save_path} (Total samples: {len(all_data)})")
        
        self.rm.release_all()
