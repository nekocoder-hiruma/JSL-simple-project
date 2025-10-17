import pickle
import cv2
import numpy as np
# Import PIL (Python Imaging Library) components for drawing Japanese text
from PIL import ImageFont, Image, ImageDraw
from constants import (
    hands_video, mp_drawing, mp_drawing_styles, mp_hands,
    MODEL_PATH, LABELS_DICT
)


def find_font_path() -> str | None:
    """
    Attempts to find a suitable font file for displaying Japanese
    characters on both Windows and macOS.
    """
    # Define potential font paths
    font_paths = [
        "C:/Windows/Fonts/YuGothM.ttc",  # Windows (Yu Gothic Medium)
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"  # macOS (Hiragino Kaku Gothic W3)
        # Add paths for Linux if needed, e.g., '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    ]

    for path in font_paths:
        try:
            # Try to load the font. If it succeeds, return the path.
            ImageFont.truetype(path, 32)
            print(f"Found font at: {path}")
            return path
        except IOError:
            # If it fails, just continue to the next path
            continue

    # If no fonts are found after checking all paths
    print("Warning: No suitable Japanese font found.")
    print("Falling back to default ASCII display.")
    return None


def run_inference() -> None:
    """
    Starts the webcam, performs real-time hand gesture detection,
    and classifies gestures using the trained model.
    """
    try:
        # Load the trained model
        with open(MODEL_PATH, "rb") as f:
            model_dict = pickle.load(f)
        model = model_dict["model"]
    except FileNotFoundError:
        print(f"Error: '{MODEL_PATH}' not found. Please train the model first.")
        return

    # Start the webcam
    cap = cv2.VideoCapture(index=1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # --- Load Font for Japanese Characters ---
    font_path = find_font_path()
    font = ImageFont.truetype(font_path, 32) if font_path else None

    # --- Define Fixed Background Box Properties ---
    BOX_WIDTH = 200  # Fixed width for the background box
    BOX_HEIGHT = 50  # Fixed height for the background box
    TEXT_PADDING = 10  # Padding for the text inside the box

    # Main loop: read from webcam frame by frame
    while True:
        # 'ret' is a boolean, 'frame' is the image array
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        H, W, _ = frame.shape
        # Convert from BGR (OpenCV) to RGB (MediaPipe)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hands
        results = hands_video.process(frame_rgb)

        # If a hand is detected
        if results.multi_hand_landmarks:
            # Use the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw the landmarks and connections on the *original* BGR frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # --- Prepare Data for Prediction ---
            landmarks_data = []  # For normalized data
            x_coords = []  # For raw x coords
            y_coords = []  # For raw y coords

            # Extract and normalize landmarks *exactly* as in dataset_creator.py
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)

            # Find the top-left-most point (min x, min y)
            base_x, base_y = min(x_coords), min(y_coords)

            # Calculate relative coordinates
            for x, y in zip(x_coords, y_coords):
                landmarks_data.append(x - base_x)
                landmarks_data.append(y - base_y)

            # --- Make Prediction ---
            # Convert landmark data to NumPy array and make a prediction
            prediction = model.predict([np.asarray(landmarks_data)])
            # Get the predicted class index (e.g., '0'), convert to int,
            # and look up the Japanese character in our dictionary
            predicted_character = LABELS_DICT[int(prediction[0])]

            # --- Draw Bounding Box and Text ---
            # Get bounding box coords from raw x and y values
            x1 = int(min(x_coords) * W) - 10
            y1 = int(min(y_coords) * H) - 10
            x2 = int(max(x_coords) * W) + 10
            y2 = int(max(y_coords) * H) + 10

            # Draw the black bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

            # --- Draw Fixed Prediction Text Box ---

            # 1. Define background box coordinates
            # Positioned above the hand's y1 coordinate
            bg_tl = (x1, y1 - BOX_HEIGHT - 20)  # Top-left corner
            bg_br = (x1 + BOX_WIDTH, y1 - 20)  # Bottom-right corner

            # 2. Define text position (padded inside the background box)
            # We'll adjust the Y position for PIL vs OpenCV due to text anchoring

            if font:
                # --- Use PIL to draw (supports Japanese) ---
                # 1. Convert to PIL Image
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                # 2. Draw white background
                draw.rectangle([bg_tl, bg_br], fill="white")

                # 3. Define text position (PIL uses top-left anchor)
                text_y = bg_tl[1] + (BOX_HEIGHT - 32) // 2  # Center vertically (approx)
                text_pos = (bg_tl[0] + TEXT_PADDING, text_y)

                # 4. Draw black text
                draw.text(text_pos, predicted_character, font=font, fill=(0, 0, 0, 255))

                # 5. Convert back to OpenCV format
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_BGR2RGB)

            else:
                # --- Fallback to OpenCV (no Japanese support) ---
                text = str(prediction[0])  # Use class index as fallback
                fontFace = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1.3
                thickness = 3

                # 1. Draw white background
                cv2.rectangle(frame, bg_tl, bg_br, (255, 255, 255), cv2.FILLED)

                # 2. Define text position (OpenCV uses bottom-left anchor)
                # Get text size to help center it
                ((text_width, text_height), baseline) = cv2.getTextSize(text, fontFace,
                                                                        fontScale, thickness)
                text_y = bg_tl[1] + (BOX_HEIGHT + text_height) // 2
                text_origin = (bg_tl[0] + TEXT_PADDING, text_y)

                # 3. Draw black text
                cv2.putText(frame, text, text_origin, fontFace, fontScale, (0, 0, 0),
                            thickness, cv2.LINE_AA)

        # Display the final frame
        cv2.imshow("frame", frame)

        # Wait 1ms for a key press. If 'q' is pressed, break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()