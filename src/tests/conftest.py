"""
Shared fixtures for the gesture recognition project.

This file provides mock objects for external dependencies
(cv2, mediapipe, tensorflow, PIL) to all test files.
Fixtures defined here are automatically discovered by pytest.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_cv2():
    """Mocks cv2 functions for create_dataset.py and helpers.py."""
    with patch('src.data.create_dataset.cv2', new_callable=MagicMock, create=True) as mock_cv2_create, \
            patch('src.utils.helpers.cv2', new_callable=MagicMock, create=True) as mock_cv2_helpers:
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_cv2_create.imread.return_value = mock_img
        mock_cv2_create.cvtColor.return_value = mock_img

        mock_cv2_helpers.imread.return_value = mock_img
        mock_cv2_helpers.cvtColor.return_value = mock_img

        yield mock_cv2_create, mock_cv2_helpers


@pytest.fixture
def mock_hands():
    """Mocks the 'hands' object from config_loader."""
    mock_hand_landmarks = MagicMock()
    mock_hand_landmarks.landmark = [MagicMock(x=0.5, y=0.5) for _ in range(21)]

    mock_results = MagicMock()
    mock_results.multi_hand_landmarks = [mock_hand_landmarks]

    with patch('src.data.create_dataset.hands', new_callable=MagicMock, create=True) as mock_hands_create, \
            patch('src.utils.helpers.hands', new_callable=MagicMock, create=True) as mock_hands_helpers:
        mock_hands_create.process.return_value = mock_results
        mock_hands_helpers.process.return_value = mock_results

        yield mock_hands_create, mock_hands_helpers, mock_results


@pytest.fixture
def mock_ml_libs():
    """Mocks TensorFlow/Keras and Scikit-learn libraries for model_trainer.py."""
    with patch('src.models.model_trainer.to_categorical', new_callable=MagicMock) as mock_to_cat, \
            patch('src.models.model_trainer.train_test_split', new_callable=MagicMock) as mock_split, \
            patch('src.models.model_trainer.Sequential', new_callable=MagicMock) as mock_sequential:
        mock_model_instance = MagicMock(spec=['fit', 'save', 'compile', 'summary'])
        mock_sequential.return_value = mock_model_instance

        yield {
            "to_categorical": mock_to_cat,
            "train_test_split": mock_split,
            "Sequential": mock_sequential,
            "model_instance": mock_model_instance
        }


@pytest.fixture
def mock_pil():
    """Mocks PIL functions for font loading and drawing in helpers.py."""
    with patch('src.utils.helpers.ImageFont', new_callable=MagicMock, create=True) as mock_image_font, \
            patch('src.utils.helpers.Image', new_callable=MagicMock, create=True) as mock_image, \
            patch('src.utils.helpers.ImageDraw', new_callable=MagicMock, create=True) as mock_image_draw:
        mock_font_instance = MagicMock()
        mock_image_font.truetype.return_value = mock_font_instance

        mock_draw_instance = MagicMock(spec=['text'])
        mock_image_draw.Draw.return_value = mock_draw_instance

        yield {
            "ImageFont": mock_image_font,
            "Image": mock_image,
            "ImageDraw": mock_image_draw,
            "font_instance": mock_font_instance,
            "draw_instance": mock_draw_instance
        }

