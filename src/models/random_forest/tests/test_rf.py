import pytest
from models.random_forest.rf_model import RFModel
from core.config_loader import load_model_config

def test_rf_model_init():
    config = load_model_config('base') # Use base for testing if specific rf config not yet defined
    model = RFModel(config)
    assert model is not None
    assert model.model is None

def test_rf_model_predict_not_trained():
    config = load_model_config('base')
    model = RFModel(config)
    with pytest.raises(ValueError):
        model.predict([0.0]*42)
