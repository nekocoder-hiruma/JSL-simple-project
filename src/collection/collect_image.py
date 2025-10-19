import os
import cv2
from src.config_loader import DATA_DIR, NUMBER_OF_CLASSES, DATASET_SIZE_PER_CLASS, LABELS_DICT


def collect_images() -> None:
    """
    Captures and saves images from the webcam for each gesture class.

    This function will:
    1. Create the main data directory if it doesn't exist.
    2. Loop through each class (0 to 19).
    3. Create a subdirectory for each class.
    4. Prompt the user to press 'S' to start capturing for that class.
    5. Capture 'DATASET_SIZE_PER_CLASS' (e.g., 100) images and save them.
    6. Repeat for the next class.
    """

    # Create the main './data' directory if it doesn't already exist.
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Use default camera (index 0). Change to 1 or higher if you have multiple cameras.
    cap = cv2.VideoCapture(index=1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Loop from 0 to NUMBER_OF_CLASSES - 1 (e.g., 0 to 19)
    for j in range(NUMBER_OF_CLASSES):
        # Create a path for the class directory (e.g., './data/0', './data/1')
        class_dir = os.path.join(DATA_DIR, str(j))

        # Create the class-specific directory if it doesn't exist.
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        # Get the corresponding Japanese character for the class number.
        char_label = LABELS_DICT[j]
        print(f"Collecting data for class {j} ('{char_label}')")

        # --- Prompt user to get ready ---
        # This loop waits for the user to press 's' before starting.
        while True:
            # Read a frame from the webcam.
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Add text to the frame to instruct the user.
            cv2.putText(frame, "Press 'S' to start!", (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            # Display the frame in a window named "frame".
            cv2.imshow("frame", frame)

            # Wait 25ms for a key press. If 's' is pressed, break the loop.
            if cv2.waitKey(25) == ord('s'):
                break

        # --- Collect images ---
        counter = 0
        while counter < DATASET_SIZE_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                break

            # Display the frame so the user can see what's being captured.
            cv2.imshow("frame", frame)
            cv2.waitKey(25)  # Short delay to allow the frame to be displayed.

            # Define the full image path (e.g., './data/0/0.jpg', './data/0/1.jpg')
            img_path = os.path.join(class_dir, f"{counter}.jpg")

            # Save the current frame as a JPG file.
            cv2.imwrite(img_path, frame)

            counter += 1

    # Clean up: release the camera and close all OpenCV windows.
    print("Data collection complete.")
    cap.release()
    cv2.destroyAllWindows()