import os
import cv2
import pickle
import csv
from typing import List, Optional
from data.collection.base_collector import BaseCollector

class StaticCollector(BaseCollector):
    """
    Captures skeletal hand landmarks for distinct static gestures.
    Implements an interactive flow where the user poses and presses 'C' to capture.
    """
    def __init__(self, config, resource_manager, mp_handler):
        super().__init__(config, resource_manager)
        self.mp_handler = mp_handler
        self.save_path = config.get_setting('paths', 'dataset_path')
        self.seq_len = config.get_int('model_params', 'sequence_length', 30)

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

    def record_sequence(self, cap, sign_text) -> Optional[List[List[float]]]:
        """Captures a short sequence of frames for a static pose."""
        sequence_data = []
        for i in range(self.seq_len):
            ret, frame = cap.read()
            if not ret: return None

            # Visual Feedback
            results = self.mp_handler.raw_process(frame)
            self.mp_handler.draw_landmarks(frame, results)
            
            # Progress HUD
            progress_text = f"Recording: {sign_text}"
            instr = f"Holding pose... ({i+1}/{self.seq_len})"
            self.draw_interactive_overlay(frame, progress_text, instructions=instr)
            
            # Recording Progress Bar
            cv2.rectangle(frame, (20, 100), (220, 110), (0, 0, 100), -1)
            fill = int(((i + 1) / self.seq_len) * 200)
            cv2.rectangle(frame, (20, 100), (20 + fill, 110), (0, 0, 255), -1)

            cv2.imshow("Capture Window", frame)
            cv2.waitKey(1)

            landmarks = self.mp_handler.process_frame(frame)
            sequence_data.append(landmarks or ([0.0] * 42))
        return sequence_data

    def collect(self):
        """Main loop to gather pose data interactively for designated characters."""
        self.prepare_directory()
        cap = self.rm.get_webcam()
        labels_dict = self.load_labels()
        classes = self.get_classes_to_collect()
        
        new_data, new_labels = [], []
        
        # Start the preview mode
        if not self.wait_for_start(cap, "Static Collection Ready"):
            self.rm.release_all()
            return

        print(f"Interactive Capture Started. Record {self.seq_len}-frame sequences for each character.")
        
        for class_idx in classes:
            char_name = labels_dict.get(class_idx, str(class_idx))
            
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Live landmark estimation and skeletal drawing
                results = self.mp_handler.raw_process(frame)
                self.mp_handler.draw_landmarks(frame, results)
                
                # Feedback HUD
                status_text = f"Pose for: {char_name}"
                instructions = "Press 'C' to Capture Sequence | 'Q' to Quit & Save"
                self.draw_interactive_overlay(frame, status_text, instructions=instructions)
                
                cv2.imshow("Capture Window", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    sequence = self.record_sequence(cap, char_name)
                    if sequence:
                        new_data.append(sequence)
                        new_labels.append(class_idx)
                        print(f"Captured sequence for: {char_name}")
                        break
                    else:
                        print("Capture failed. Try again.")
                
                if key == ord('q'):
                    print("Quitting and saving current progress...")
                    break
            
            if key == ord('q'): break

        # Incremental saving logic
        if new_data:
            all_data, all_labels = [], []
            if os.path.exists(self.save_path):
                print(f"Loading existing dataset from {self.save_path}...")
                with open(self.save_path, "rb") as f:
                    old_dataset = pickle.load(f)
                    all_data = old_dataset.get('data', [])
                    all_labels = old_dataset.get('labels', [])
            
            all_data.extend(new_data)
            all_labels.extend(new_labels)
            
            with open(self.save_path, "wb") as f:
                pickle.dump({"data": all_data, "labels": all_labels}, f)
            print(f"Dataset updated successfully at {self.save_path} (Total samples: {len(all_data)})")

        self.rm.release_all()
