import mediapipe as mp

# --- Dirs and Paths ---
# DATA_DIR specifies the root directory where collected images will be stored.
DATA_DIR = "./data"
# DATASET_PATH is the filename for the processed dataset (landmarks and labels).
DATASET_PATH = "data.pickle"
# MODEL_PATH is the filename for the trained machine learning model.
OLD_MODEL_PATH = "model.p"
MODEL_PATH = "jsl_model.keras" # Keras model uses .h5 or .keras

# --- New constants for sequence modeling ---
# SEQUENCE_LENGTH is the number of frames to capture for one sign.
SEQUENCE_LENGTH = 30

# --- Model and Dataset Config ---
# DATASET_SIZE_PER_CLASS defines how many images to capture for each gesture.
DATASET_SIZE_PER_CLASS = 20

"""
--- Japanese Sign Language Labels (あ to と) ---
This dictionary maps the integer class index (0, 1, 2...)
to the corresponding Japanese character. This is used in the inference
script to show the human-readable prediction.
"""
LABELS_DICT = {
    0: 'あ', 1: 'い', 2: 'う', 3: 'え', 4: 'お',
    5: 'か', 6: 'き', 7: 'く', 8: 'け', 9: 'こ',
    10: 'さ', 11: 'し', 12: 'す', 13: 'せ', 14: 'そ',
    15: 'た', 16: 'ち', 17: 'つ', 18: 'て', 19: 'と',
    20: 'な', 21: 'に', 22: 'ぬ', 23: 'ね', 24: 'の',
    25: 'は', 26: 'ひ', 27: 'ふ', 28: 'へ', 29: 'ほ',
    30: 'ま', 31: 'み', 32: 'む', 33: 'め', 34: 'も'
}

# NUMBER_OF_CLASSES defines how many different gestures we want to train (あ to と is 20).
NUMBER_OF_CLASSES = len(LABELS_DICT)

# --- MediaPipe Setup ---
# Import the specific MediaPipe solutions we need.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

"""
Initialize the MediaPipe Hands solution.
'hands' is used for processing static images (dataset creation).
- static_image_mode=True: Optimizes for individual images (not a video stream).
- max_num_hands=1: We are only tracking one hand for these signs.
- min_detection_confidence=0.5: The minimum confidence (50%) for a hand
  detection to be considered successful.
"""
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.5)

"""
'hands_video' is used for processing the real-time video stream (inference).
- static_image_mode=False: Optimizes for a video stream.
- min_tracking_confidence=0.5: The minimum confidence (50%) for the hand
  landmarks to be successfully tracked from one frame to the next.
"""
hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.5, min_tracking_confidence=0.5)