"""
Pytest unit tests for the `model_trainer.py` module.
Tests are grouped by function into separate classes.
"""

import pickle
from unittest.mock import patch, MagicMock, mock_open

import numpy as np

# --- Import functions to be tested ---
from src.models.model_trainer import (
    load_dataset,
    prepare_data,
    build_lstm_model,
    train_and_save_model,
    main_trainer
)


# --- Tests for load_dataset ---

class TestLoadDataset:

    @patch('builtins.open', new_callable=mock_open, read_data=pickle.dumps({"data": [[1]], "labels": [0]}))
    @patch('src.models.model_trainer.to_categorical', return_value=np.array([[1, 0]]))
    def test_positive_load_successful(self, mock_to_cat, mock_file):
        """
        Positive test: Checks successful loading and processing of a pickle file.
        """
        # Mock config values
        with patch('src.models.model_trainer.NUMBER_OF_CLASSES', 2):
            result_data, result_labels = load_dataset("dummy.pickle")

        mock_file.assert_called_with("dummy.pickle", "rb")

        assert np.array_equal(result_data, np.array([[1]]))

        mock_to_cat.assert_called_with(np.array([0]), num_classes=2)
        assert np.array_equal(result_labels, np.array([[1, 0]]))

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_negative_file_not_found(self, mock_file):
        """
        Negative test: Checks behavior when the pickle file is not found.
        """
        result = load_dataset("non_existent.pickle")
        assert result is None
        assert result != (None, None)  # Check against a different falsey value


# --- Tests for prepare_data ---

class TestPrepareData:

    def test_positive_split_successful(self, mock_ml_libs):
        """
        Positive test: Checks that data is passed to train_test_split correctly.
        """
        data = np.array([[1], [2], [3]])
        labels = np.array([0, 1, 0])

        mock_split = mock_ml_libs["train_test_split"]
        mock_split.return_value = ("x_train", "x_test", "y_train", "y_test")

        result = prepare_data(data, labels)

        # Check that train_test_split was called with correct args
        mock_split.assert_called_once()
        assert np.array_equal(mock_split.call_args[0][0], data)
        assert np.array_equal(mock_split.call_args[0][1], labels)
        assert np.isclose(mock_split.call_args[1]['test_size'], 0.2, rtol=1e-09, atol=1e-09)
        assert mock_split.call_args[1]['shuffle'] is True

        assert result == ("x_train", "x_test", "y_train", "y_test")


# --- Tests for build_lstm_model ---

class TestBuildLSTMModel:

    def test_positive_model_built_and_compiled(self, mock_ml_libs):
        """
        Positive test: Checks if the Keras Sequential model is built and compiled.
        """
        mock_sequential = mock_ml_libs["Sequential"]
        mock_model_instance = mock_ml_libs["model_instance"]

        # Mock config values
        with patch('src.models.model_trainer.NUMBER_OF_CLASSES', 5), \
                patch('src.models.model_trainer.SEQUENCE_LENGTH', 10):
            model = build_lstm_model()

        assert model == mock_model_instance

        mock_model_instance.compile.assert_called_with(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        mock_model_instance.summary.assert_called_once()
        mock_sequential.assert_called_once()


# --- Tests for train_and_save_model ---

class TestTrainAndSaveModel:

    def test_positive_train_and_save(self, mock_ml_libs):
        """
        Positive test: Checks if model.fit and model.save are called.
        """
        mock_model_instance = mock_ml_libs["model_instance"]

        with patch('src.models.model_trainer.MODEL_PATH', 'test_model.h5'):
            train_and_save_model(
                mock_model_instance,
                "x_train", "y_train", "x_test", "y_test"
            )

        mock_model_instance.fit.assert_called_with(
            "x_train", "y_train",
            epochs=50,
            validation_data=("x_test", "y_test")
        )
        mock_model_instance.save.assert_called_with('test_model.h5')


# --- Tests for main_trainer ---

class TestMainTrainer:

    @patch('src.models.model_trainer.load_dataset', return_value=(np.array([1]), np.array([0])))
    @patch('src.models.model_trainer.prepare_data', return_value=("x_train", "x_test", "y_train", "y_test"))
    @patch('src.models.model_trainer.build_lstm_model')
    @patch('src.models.model_trainer.train_and_save_model')
    @patch('src.models.model_trainer.DATASET_PATH', 'mock_dataset.pickle')
    def test_positive_orchestrator(self, mock_train, mock_build, mock_prepare, mock_load):
        """
        Positive integration test for the main_trainer orchestrator.
        """
        mock_model = MagicMock()
        mock_build.return_value = mock_model

        main_trainer()

        mock_load.assert_called_with('mock_dataset.pickle')
        mock_prepare.assert_called_with(np.array([1]), np.array([0]))
        mock_build.assert_called_once()
        mock_train.assert_called_with(mock_model, "x_train", "y_train", "x_test", "y_test")

    @patch('src.models.model_trainer.load_dataset', return_value=None)
    @patch('src.models.model_trainer.prepare_data')
    @patch('src.models.model_trainer.build_lstm_model')
    def test_negative_load_fail(self, mock_build, mock_prepare, mock_load):
        """
        Negative test: Checks that the pipeline stops if loading fails.
        """
        main_trainer()

        mock_load.assert_called_once()
        mock_prepare.assert_not_called()
        mock_build.assert_not_called()

