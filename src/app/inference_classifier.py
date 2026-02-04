import cv2
import numpy as np
import mediapipe as mp
from core.config_loader import load_specialized_config
from core.resource_manager import ResourceManager
from core.mediapipe_utils import MediaPipeHandler
from models.random_forest.rf_model import RFModel
from utils.helpers import draw_visual_feedback

def run_inference():
    """
    Main loop for recognizing static signs in real-time.
    Loads the model, opens the webcam, and provides live feedback on the screen.
    """
    print("Running random forest model inference.")
    config = load_specialized_config('rf')
    rm = ResourceManager()
    handler = MediaPipeHandler(is_video_stream=True)
    
    # Initialize model and load its previous training
    model = RFModel(config)
    try:
        model.load()
    except Exception:
        print("Could not load a trained model. Please train one first.")
        return

    cap = rm.get_webcam()
    font = rm.get_japanese_font()
    
    # Shared labels for displaying recognized signs
    import csv
    labels_dict = {}
    labels_path = config.get_setting('paths', 'labels_path')
    with open(labels_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row: labels_dict[int(row[0])] = row[1]

    print("Live Recognition Started. Press 'Q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Detect hand and extract position data
        points = handler.process_frame(frame)
        if points:
            # Get the model's guess and confidence level
            result_idx, confidence = model.predict(points)
            sign_name = f"Predicted character: {labels_dict.get(int(result_idx), 'Unknown')} ({confidence*100:.1f}%)"
            
            # Display results to the user
            draw_visual_feedback(frame, sign_name, (50, 50), font)

        cv2.imshow("Sign Language Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rm.release_all()
    handler.close()