import cv2
import numpy as np
from collections import deque
from core.config_loader import load_specialized_config
from core.resource_manager import ResourceManager
from core.mediapipe_utils import MediaPipeHandler
from models.lstm.lstm_model import LSTMModel
from utils.helpers import draw_visual_feedback

def run_inference():
    """
    Main loop for recognizing dynamic motion-based signs in real-time.
    Continuously buffers hand movement data and uses the LSTM model to identify signatures.
    """
    config = load_specialized_config('keras')
    rm = ResourceManager()
    handler = MediaPipeHandler(is_video_stream=True)
    
    # Configuration parameters
    seq_len = config.get_int('model_params', 'sequence_length', 30)
    confidence_threshold = config.get_float('inference', 'confidence_threshold', 0.5)
    
    # Initialize model and motion buffer
    model = LSTMModel(config)
    try:
        model.load()
    except Exception:
        print("Could not load a trained sequence model. Please train one first.")
        return

    buffer = deque(maxlen=seq_len)
    cap = rm.get_webcam()
    font = rm.get_japanese_font()
    
    # Load labels for human-readable output
    import csv
    labels_dict = {}
    labels_path = config.get_setting('paths', 'labels_path')
    with open(labels_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row: labels_dict[int(row[0])] = row[1]

    print("Motion Recognition Started. Press 'Q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Collect hand positions into the rolling buffer
        points = handler.process_frame(frame)
        buffer.append(points or ([0.0] * 42))

        # Once the buffer is full, attempt a prediction
        if len(buffer) == seq_len:
            result_idx, confidence = model.predict(list(buffer))
            sign_name = f"Predicted character: {labels_dict.get(result_idx, 'Unknown')} ({confidence*100:.1f}%)"
            
            # Label the video stream with the detected sign
            draw_visual_feedback(frame, sign_name, (50, 50), font)

        cv2.imshow("Sign Movement Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rm.release_all()
    handler.close()