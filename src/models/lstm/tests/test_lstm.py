import pytest
from models.lstm.lstm_model import LSTMModel
from core.config_loader import load_model_config

def test_lstm_model_init():
    config = load_model_config('keras')
    model = LSTMModel(config)
    assert model is not None
    assert model.model is None

def test_lstm_model_predict_not_trained():
    config = load_model_config('keras')
    model = LSTMModel(config)
    with pytest.raises(ValueError):
        model.predict([[0.0]*42]*30)
