# Real-Time Japanese Sign Language (JSL) Gesture Recognizer

This project is a real-time hand gesture recognition system built to classify the first 20 characters of the Japanese hiragana syllabary using a standard webcam in the first iteration. It leverages computer vision and machine learning to create a flexible and robust pipeline for data collection, model training, and inference.

The project has evolved to a LSM to recognise motion based gesture and optimization development is still in progress.

This project was inspired by a tutorial from the [Computer Vision Engineer YouTube channel](https://www.youtube.com/watch?v=MJCSjXepaAM) and has been significantly evolved to support dynamic gestures and a more professional development workflow.


## Features

* **Dynamic Gesture Recognition**: Utilizes an LSTM deep learning model to understand signs that involve motion over time.
* **Centralized Configuration**: All settings are managed in a simple `config.ini` file, allowing for easy changes without touching the code.
* **Advanced Data Collection**: A highly user-friendly and flexible data collection script that supports:
    * **Continuous Collection Mode**: Automatically captures all samples for a sign after a single prompt.
    * **Resumable Sessions**: Stop at any point, and the script saves your progress. Start again, and it appends to your existing dataset.
    * **Start From Any Character**: Configure which sign you want to start collecting from.
    * **Review and Discard**: Immediately review and discard any sample you're not happy with.
* **Modular and Maintainable Code**: Shared logic is centralized in a `helpers.py` file, following the DRY (Don't Repeat Yourself) principle.

## Configuration

Before running the project, you can adjust settings in the **`config.ini`** file. This is where you can set the camera you want to use, change file paths, or modify model parameters.

**Key Settings to Note:**

* `camera/index`: Set this to `0` for your built-in webcam, or `1`, `2`, etc., for external cameras.
* `data_collection/start_class_index`: Set this to the character index you want to start collecting from (e.g., `15` for 'た'). The script will resume from this point.

## Setup and Installation

### Prerequisites

* Python 3.12+
* [uv](https://github.com/astral-sh/uv) installed (`pip install uv`).
* A webcam connected to your computer.

### 1. Set Up the Environment with `uv`

This project uses `uv`, a fast Python package installer and resolver.

```bash
# Create and activate a virtual environment
uv venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

Install all the required Python libraries from the uv lock file:

```bash
uv sync
```
## How to Run the Project

The project workflow is divided into three main scripts. You must run them in the following order.

### Step 1: Collect Sequence Data

This script launches an advanced data collection interface. It will automatically load any existing `data.pickle` file and allow you to add to it.

1.  **Configure `config.ini`:** Open `config.ini` and set `start_class_index` to the character you wish to start from.
2.  **Run the collection script:**
    ```bash
    python main.py collect
    ```
3.  A window will appear, prompting you to press **'S'** to begin collecting all samples for the starting character.
4.  The script will then automatically record each sample, with a countdown timer in between to allow you to reset.
5.  After each recording, you have a moment to press **'D'** to discard and re-record the last sample.
6.  Once all samples for a character are done, you will be prompted for the next one. You can press **'Q'** at this screen to quit and save all your progress.

### Step 2: Train the Model

This script loads the `data.pickle` file, trains the LSTM model, and saves the trained model as `jsl_model.h5`.

```bash
python main.py train
```
### Step 3: Run Real-Time Inference

You're all set! This final step runs the real-time gesture recognition.

1.  Run the inference script:
    ```bash
    python main.py inference
    ```
## Legacy Version (RandomForest Model)

The original version of this project, which uses a `RandomForestClassifier` for **static** gesture recognition, is available in the `original_randomforest` directory. This version is faster and may have higher accuracy for static poses but does not support dynamic gestures.

## File Structure

The project is organized into a src package to ensure modularity and scalability. All operations are run from within this directory.
```aiignore
JSL-simple-project/
└── src/
    ├── app/
    │   └── inference.py
    ├── collection/
    │   └── collect_sequences.py
    ├── configs/
    │   └── config.ini
    ├── data/
    │   └── (functions to create dataset for model training)
    ├── dataset/
    │   └── (raw collected data and processed data appears here)
    ├── labels/
    │   └── labels.csv
    ├── models/
    │   └── model_trainer.py
    ├── saved_models/
    │   ├── jsl_model.keras
    │   └── model.p
    ├── tests/
    │   └── ...
    ├── utils/
    │   └── helpers.py
    ├── __init__.py
    ├── config_loader.py
    ├── main.py
    └── rf_main.py

```