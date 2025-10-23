"""
Pytest unit tests for the `create_dataset.py` module.
Tests are grouped by function into separate classes.
"""

import os
from unittest.mock import patch, mock_open, call

import pytest
# --- Import functions to be tested ---
from src.data.create_dataset import process_image, save_dataset, create_dataset


# --- Tests for process_image ---

class TestProcessImage:
    def test_positive_landmarks_normalized(self, mock_cv2, mock_hands):
        """
        Positive test: Checks if landmarks are correctly processed and normalized.
        """
        mock_cv2_create, _ = mock_cv2
        mock_hands_create, _, _ = mock_hands

        min_x = 0.5
        min_y = 0.7
        # Modify landmarks to test normalization
        mock_hand = mock_hands_create.process.return_value.multi_hand_landmarks[0]
        mock_hand.landmark[0].x = 0.6
        mock_hand.landmark[0].y = 0.8
        mock_hand.landmark[1].x = min_x  # This is the min_x
        mock_hand.landmark[1].y = min_y  # This is the min_y

        for i in range(2, 21):  # Set all other landmarks to min
            mock_hand.landmark[i].x = min_x
            mock_hand.landmark[i].y = min_y

        result = process_image("dummy_path.jpg")

        # Check if normalization was applied
        assert result is not None
        assert len(result) == 42  # 21 * 2
        assert result[0] == pytest.approx(0.6 - min_x)
        assert result[1] == pytest.approx(0.8 - min_y)
        assert result[2] == pytest.approx(0.5 - min_x)
        assert result[3] == pytest.approx(0.7 - min_y)

        # Check that cv2 functions were called
        mock_cv2_create.imread.assert_called_with("dummy_path.jpg")
        mock_cv2_create.cvtColor.assert_called_once()

    def test_negative_no_hand_detected(self, mock_cv2, mock_hands):
        """
        Negative test: Checks behavior when no hand is detected.
        """
        _, _, mock_results = mock_hands
        mock_results.multi_hand_landmarks = None  # Set mock to return no landmarks

        result = process_image("no_hand.jpg")

        assert result is None
        assert result != []  # Check against a different falsey value

    def test_negative_image_read_fail(self, mock_cv2, mock_hands):
        """
        Negative edge case: Checks behavior when cv2.imread fails.
        """
        mock_cv2_create, _ = mock_cv2
        mock_hands_create, _, _ = mock_hands

        mock_cv2_create.imread.return_value = None  # Simulate corrupt file

        result = process_image("corrupt.jpg")

        assert result is None
        mock_hands_create.process.assert_not_called()  # Check that processing stopped


# --- Tests for save_dataset ---

class TestSaveDataset:

    @patch('src.data.create_dataset.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_positive_save_successful(self, mock_file, mock_pickle_dump):
        """
        Positive test: Checks if data is correctly saved to a pickle file.
        """
        data = [[1.0, 2.0], [3.0, 4.0]]
        labels = ["A", "B"]
        filepath = "test.pickle"

        save_dataset(data, labels, filepath)

        mock_file.assert_called_with(filepath, "wb")
        expected_data = {"data": data, "labels": labels}
        mock_pickle_dump.assert_called_with(expected_data, mock_file())

    @patch('src.data.create_dataset.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_negative_empty_data(self, mock_file, mock_pickle_dump):
        """
        Negative test: Checks that nothing is saved if data is empty.
        """
        save_dataset([], [], "empty.pickle")

        mock_file.assert_not_called()
        mock_pickle_dump.assert_not_called()

    @patch('src.data.create_dataset.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_negative_io_error(self, mock_file, mock_pickle_dump):
        """
        Negative edge case: Checks handling of an IOError during saving.
        """
        mock_file.side_effect = IOError("Disk full")

        data = [[1.0, 2.0]]
        labels = ["A"]
        filepath = "test.pickle"

        save_dataset(data, labels, filepath)  # Function should catch error and print

        mock_file.assert_called_with(filepath, "wb")
        mock_pickle_dump.assert_not_called()  # Fails before dump


# --- Tests for create_dataset ---

class TestCreateDataset:

    @patch('src.data.create_dataset.os.path.exists', return_value=True)
    @patch('src.data.create_dataset.os.listdir')
    @patch('src.data.create_dataset.os.path.isdir')
    @patch('src.data.create_dataset.process_image')
    @patch('src.data.create_dataset.save_dataset')
    @patch('src.data.create_dataset.DATA_DIR', 'mock_data_dir')  # Mock config constant
    @patch('src.data.create_dataset.DATASET_PATH', 'mock_dataset.pickle')  # Mock config constant
    def test_positive_orchestrator(self, mock_save, mock_process, mock_isdir, mock_listdir, mock_exists):
        """
        Positive integration test for the main create_dataset orchestrator.
        """
        # Mock file system structure
        mock_listdir.side_effect = [
            ['class_A', 'class_B', 'not_a_dir.txt'],  # Top level
            ['img1.jpg', 'img2.jpg'],  # Inside class_A
            ['img3.jpg']  # Inside class_B
        ]
        mock_isdir.side_effect = [True, True, False, True, True, True]

        # Mock image processing results
        mock_process.side_effect = [[1.1], [1.2], [2.1]]  # Landmarks for img1, img2, img3

        create_dataset()

        # Check that os.path.exists was called
        mock_exists.assert_called_with('mock_data_dir')

        # Check that process_image was called for each image
        expected_calls = [
            call(os.path.join('mock_data_dir', 'class_A', 'img1.jpg')),
            call(os.path.join('mock_data_dir', 'class_A', 'img2.jpg')),
            call(os.path.join('mock_data_dir', 'class_B', 'img3.jpg')),
        ]
        assert mock_process.call_count == 3
        mock_process.assert_has_calls(expected_calls)

        # Check that save_dataset was called with aggregated data
        expected_data = [[1.1], [1.2], [2.1]]
        expected_labels = ['class_A', 'class_A', 'class_B']
        mock_save.assert_called_with(expected_data, expected_labels, 'mock_dataset.pickle')

    @patch('src.data.create_dataset.os.path.exists', return_value=False)
    @patch('src.data.create_dataset.os.listdir')
    @patch('src.data.create_dataset.save_dataset')
    @patch('src.data.create_dataset.DATA_DIR', 'mock_data_dir')
    def test_negative_data_dir_not_found(self, mock_save, mock_listdir, mock_exists):
        """
        Negative test: Checks that the function exits early if DATA_DIR is not found.
        """
        create_dataset()

        mock_exists.assert_called_with('mock_data_dir')
        mock_listdir.assert_not_called()
        mock_save.assert_not_called()

