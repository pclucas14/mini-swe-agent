from minisweagent.models import get_model_class
from minisweagent.models.trapi_model import TrapiModel


def test_get_model_class_trapi():
    """Test that get_model_class returns TrapiModel for TRAPI model names."""
    assert get_model_class("gpt-4o_2024-11-20") == TrapiModel
    assert get_model_class("trapi-model") == TrapiModel
    assert get_model_class("model-with-gcr/preview") == TrapiModel


def test_trapi_in_model_name():
    """Test that model names containing 'trapi' use TrapiModel."""
    assert get_model_class("my-trapi-model") == TrapiModel
    assert get_model_class("TRAPI_MODEL_NAME") == TrapiModel
