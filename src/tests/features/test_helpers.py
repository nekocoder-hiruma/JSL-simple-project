"""
Pytest unit tests for the `helpers.py` module.
Tests are grouped by function into separate classes.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call

# --- Import functions to be tested ---
from src.utils.helpers import find_font_path, draw_text, process_frame


# --- Tests for find_font_path ---

class TestFindFontPath:

    @patch('src.utils.helpers.ImageFont.truetype')
    def test_positive_found_windows(self, mock_truetype):
        """
        Positive test: Font is found on the first try (Windows).
        """
        mock_truetype.return_value = MagicMock()  # Simulate successful load

        expected_path = "C:/Windows/Fonts/YuGothM.ttc"
        result = find_font_path()

        assert result == expected_path
        mock_truetype.assert_called_with(expected_path, 32)

    @patch('src.utils.helpers.ImageFont.truetype')
    def test_positive_found_macos(self, mock_truetype):
        """
        Positive test: Font is found on the second try (macOS).
        """
        mock_truetype.side_effect = [IOError, MagicMock()]  # Fail on first, succeed on second

        win_path = "C:/Windows/Fonts/YuGothM.ttc"
        mac_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
        result = find_font_path()

        assert result == mac_path
        assert mock_truetype.call_count == 2
        mock_truetype.assert_has_calls([call(win_path, 32), call(mac_path, 32)])

    @patch('src.utils.helpers.ImageFont.truetype', side_effect=IOError)
    def test_negative_font_not_found(self, mock_truetype):
        """
        Negative test: No font is found after checking all paths.
        """
        result = find_font_path()
        assert result is None
        assert mock_truetype.call_count == 2  # Tried all listed paths


# --- Tests for draw_text ---

class TestDrawText:

    def test_positive_with_font(self, mock_cv2, mock_pil):
        """
        Positive test: Drawing text with a valid font uses the PIL pipeline.
        """
        _, mock_cv2_helpers = mock_cv2

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        font = mock_pil["font_instance"]
        draw_instance = mock_pil["draw_instance"]

        draw_text(frame, "Test", (10, 10), font, color=(255, 0, 0))  # BGR color

        # Check that PIL pipeline was used
        mock_pil["Image"].fromarray.assert_called_once()
        mock_pil["ImageDraw"].Draw.assert_called_once()

        # Check that text was drawn with correct (RGB) color
        draw_instance.text.assert_called_with((10, 10), "Test", font=font, fill=(0, 0, 255))  # RGB

        # Check that OpenCV fallback was NOT used
        mock_cv2_helpers.putText.assert_not_called()

    def test_negative_font_is_none(self, mock_cv2, mock_pil):
        """
        Negative test: Fallback to cv2.putText when font is None.
        """
        _, mock_cv2_helpers = mock_cv2

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        draw_text(frame, "Test", (10, 10), font=None, color=(255, 0, 0))

        # Check that PIL pipeline was NOT used
        mock_pil["Image"].fromarray.assert_not_called()
        mock_pil["ImageDraw"].Draw.assert_not_called()

        # Check that OpenCV fallback WAS used
        mock_cv2_helpers.putText.assert_called_with(
            frame, "Font not found", (10, 10),
            mock_cv2_helpers.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, mock_cv2_helpers.LINE_AA
        )


# --- Tests for process_frame ---

class TestProcessFrame:

    def test_positive_landmarks_normalized(self, mock_cv2, mock_hands):
        """
        Positive test: Checks frame processing and landmark normalization.
        """
        _, mock_cv2_helpers = mock_cv2
        _, mock_hands_helpers, _ = mock_hands

        mock_hand = mock_hands_helpers.process.return_value.multi_hand_landmarks[0]
        mock_hand.landmark[0].x = 1.2
        mock_hand.landmark[0].y = 0.8
        mock_hand.landmark[1].x = 1.0  # min_x
        mock_hand.landmark[1].y = 0.5  # min_y

        for i in range(2, 21):  # Set all others to min
            mock_hand.landmark[i].x = 1.0
            mock_hand.landmark[i].y = 0.5

        result = process_frame(np.zeros((100, 100, 3)))

        assert result is not None
        assert len(result) == 42
        assert result[0] == pytest.approx(1.2 - 1.0)
        assert result[1] == pytest.approx(0.8 - 0.5)
        assert result[2] == pytest.approx(1.0 - 1.0)
        assert result[3] == pytest.approx(0.5 - 0.5)

    def test_negative_no_hand_detected(self, mock_cv2, mock_hands):
        """
        Negative test: No hand detected in the frame.
        """
        _, _, mock_results = mock_hands
        mock_results.multi_hand_landmarks = None  # No hands

        result = process_frame(np.zeros((100, 100, 3)))

        assert result is None

    def test_negative_processing_exception(self, mock_cv2, mock_hands):
        """
        Negative edge case: An unexpected exception occurs during processing.
        """
        _, mock_cv2_helpers = mock_cv2

        # Simulate an error during color conversion
        mock_cv2_helpers.cvtColor.side_effect = Exception("Test exception")

        result = process_frame(np.zeros((100, 100, 3)))

        assert result is None

