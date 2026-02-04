# Real-Time Japanese Sign Language (JSL) Gesture Recognizer

This project is a real-time hand gesture recognition system built to study ML models for Japanese sign language recognition. It supports both static signs (individual positions) and dynamic gestures (motions over time) using MediaPipe for landmark extraction.
This project was inspired by a tutorial from the [Computer Vision Engineer YouTube channel](https://www.youtube.com/watch?v=MJCSjXepaAM).

## Features

*   **Interactive Interface**: A unified entry point guides you through data collection, training, and inference via easy-to-follow prompts.
*   **Static & Dynamic Recognition**:
    *   **Random Forest**: Fast and accurate for static signs (single photos).
    *   **LSTM (RNN)**: Analyzes sequences of motion for dynamic signs.
*   **Modular Architecture**: Clean separation between core logic, data processing, and model implementations.
*   **Centralized Resource Management**: Efficient handling of webcam and font resources to avoid duplication.
*   **Tiered Configuration**: Load base project settings and override them with model-specific configurations.

## Setup and Installation

### Prerequisites

*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) installed (`pip install uv`).
*   A webcam connected to your computer.

### 1. Set Up the Environment

This project uses `uv` for lightning-fast dependency management.

```bash
# Create and activate a virtual environment
uv venv

# Activate the environment (macOS/Linux)
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv sync
```

## How to Run the Project

Everything is controlled via a single interactive script:

```bash
python src/main.py
```

### Workflow

1.  **Gather Training Data**:
    *   Choose **Photo Capture** for static signs or **Motion Recording** for dynamic sequences.
    *   Follow the on-screen prompts (e.g., Press 'S' to start) to capture data for each category.
2.  **Train Decision Models**:
    *   Select the model type (**Random Forest** for static or **LSTM** for dynamic).
    *   The model will learn from your collected data and save its progress in `src/saved_models/`.
3.  **Run Live Gesture Recognition**:
    *   Select your desired recognition mode.
    *   The webcam will open, providing real-time feedback and labeling the detected signs on your screen.

## Configuration

Settings are managed in the `src/configs/` directory:
*   `base_config.ini`: General project paths and camera settings.
*   `rf_config.ini`: Specific parameters for the Random Forest model.
*   `keras_config.ini`: Specific parameters for the LSTM model.

## File Structure

The project follows a modular refined structure within the `src` package:

```text
src/
├── core/               # Configuration, Resource management, and MediaPipe logic
├── data/               # Collection scripts (Static/Dynamic) and Processing utilities
├── models/             # Self-contained model implementations (Base, RF, LSTM)
├── app/                # Real-time inference applications
├── utils/              # Shared helper functions for visual feedback
├── configs/            # Config files (.ini)
├── saved_models/       # Directory for trained model artifacts
├── labels/             # Mapping of category indices to Japanese characters
└── main.py             # Interactive project controller
```