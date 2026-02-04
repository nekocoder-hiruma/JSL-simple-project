import os
import cv2
from data.collection.base_collector import BaseCollector

class StaticCollector(BaseCollector):
    """
    Captures individual photos to represent distinct hand gestures.
    Saves raw images into organized folders for later processing.
    """
    def collect(self):
        """Main loop to gather photo data for all defined gesture categories."""
        self.prepare_directory()
        cap = self.rm.get_webcam()

        for class_idx in range(self.num_classes):
            class_dir = os.path.join(self.data_dir, str(class_idx))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            prompt = f"Sign for Label {class_idx}: Press 'S' to Start"
            if not self.wait_for_start(cap, prompt):
                break

            count = 0
            while count < self.samples_per_class:
                ret, frame = cap.read()
                if not ret: break

                cv2.imshow("Capture Window", frame)
                cv2.waitKey(25)

                image_name = f"{count}.jpg"
                cv2.imwrite(os.path.join(class_dir, image_name), frame)
                count += 1

        self.rm.release_all()
