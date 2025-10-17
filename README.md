# Real-Time Japanese Sign Language (JSL) Gesture Recognition

This project is a real-time hand gesture recognition system built to classify the first 20 characters of the Japanese hiragana syllabary using a standard webcam. It leverages computer vision and machine learning to create a simple yet effective pipeline for data collection, model training, and inference.



## Features

* **Real-Time Classification**: Identifies JSL gestures from a live webcam feed.
* **Simple Training Pipeline**: A command-line interface to easily collect data, create a dataset, train the model, and run inference.
* **Lightweight Model**: Uses a `RandomForestClassifier` from Scikit-learn, which is fast and efficient for this task.
* **Cross-Platform**: Designed to work on both Windows and macOS.

## Characters Covered

This model is trained to recognize the following 20 static hiragana gestures:

* **あ, い, う, え, お** (a, i, u, e, o)
* **か, き, く, け, こ** (ka, ki, ku, ke, ko)
* **さ, し, す, せ, そ** (sa, shi, su, se, so)
* **た, ち, つ, て, と** (ta, chi, tsu, te, to)

**Note:** This implementation is designed for static hand poses and does not currently support gestures that require motion.

---

## Setup and Installation

### Prerequisites

* Python 3.12+
* [uv](https://github.com/astral-sh/uv) installed (`pip install uv`).
* A webcam connected to your computer.

### 1. Set Up the Environment with `uv`
_________

This project uses `uv`, a fast Python package installer and resolver, written in Rust.

```bash
# Create and activate a virtual environment using uv
# This creates a .venv folder in your project root.
uv venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies
______

Install all the required Python libraries using uv pip:
```Bash

uv pip install opencv-python mediapipe scikit-learn numpy pillow
```

## How to Run the Project
_______________

The project workflow is managed by main.py and is divided into four simple steps. You must run them in the following order.

Step 1: Collect Image Data

This step launches your webcam and guides you through collecting 100 images for each of the 20 gestures.

Run the collection script: `python main.py collect`

A window will appear showing your webcam feed.

The console will prompt you to collect data for class 0 ('あ').

Position your hand to make the sign for 'あ'.

Press the 'S' key to begin capturing 100 images for that sign.

The system will automatically proceed to the next class (class 1 ('い')) and wait for you to press 'S' again.

Repeat this process for all 20 characters. The collected images will be saved in the data/ directory.

Step 2: Create the Dataset

This script processes all the images you collected, extracts hand landmarks using MediaPipe, normalizes the data, and saves it into a single data.pickle file.

`python main.py dataset`

Step 3: Train the Model

This script loads the data.pickle file, trains a RandomForest classifier on the landmark data, and saves the trained model as "model.p".

`python main.py train`

Step 4: Run Real-Time Inference

You're all set! This final step runs the real-time gesture recognition.

Run the inference script: `python main.py inference`

A window will appear with your webcam feed.

Show one of the 20 trained JSL gestures to the camera.

The model will draw a bounding box around your hand and display the predicted hiragana character above it.

To stop the program, press the 'q' key.

## File Structure
____________

`main.py`: The main entry point to run any part of the project.

`constants.py`: Contains all project-wide constants, such as file paths and character labels.

`collect_image.py`: Logic for capturing and saving gesture images.

`create_dataset.py`: Logic for processing images and creating the landmark dataset.

`train_classifier.py`: Logic for training the machine learning model.

`inference_classifier.py`: Logic for running the real-time classification using the webcam.

### Important Notes
_____________

**Camera Index**: The scripts default to using camera index 0. If you have multiple cameras, you may need to change cv2.VideoCapture(0) to cv2.VideoCapture(1) (or higher) in data_collection.py and inference.py.

**Font Dependency**: To display Japanese characters correctly, the inference.py script attempts to locate system fonts. Ensure you have standard Japanese fonts installed on your Windows or macOS system. If no font is found, it will fall back to displaying the class number (0-19).


## Credits
_______

This project draws inspiration and foundational knowledge from the excellent tutorials by Computer vision engineer: https://www.youtube.com/watch?v=MJCSjXepaAM