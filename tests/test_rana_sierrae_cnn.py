import pytest
import sys
from pathlib import Path

import pandas as pd

from bioacoustics_model_zoo.rana_sierrae_cnn import RanaSierraeCNN


@pytest.fixture
def model():
    return RanaSierraeCNN()


@pytest.fixture
def train_df():
    df = pd.DataFrame(
        {
            "file": ["tests/data/birds_10s.wav"] * 3,
            "start_time": [0, 3, 6],
            "end_time": [3, 6, 9],
            "rana_sierrae": [1, 0, 1],
            "negative": [0, 1, 0],
        }
    ).set_index(["file", "start_time", "end_time"])

    return df


def test_rana_sierrae_cnn_init(model):
    """Test that Rana Sierrae CNN initializes correctly."""
    assert hasattr(model, 'classes')
    assert hasattr(model, 'preprocessor')
    assert hasattr(model, 'network')


@pytest.mark.skipif(not Path("tests/data/birds_10s.wav").exists(),
                   reason="Test audio file not available")
def test_rana_sierrae_cnn_predict(model):
    """Test Rana Sierrae CNN prediction method."""
    predictions = model.predict(["tests/data/birds_10s.wav"], batch_size=2)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) > 0


@pytest.mark.skipif(not Path("tests/data/birds_10s.wav").exists(),
                   reason="Test audio file not available")
def test_rana_sierrae_cnn_embed(model):
    """Test Rana Sierrae CNN embedding method."""
    embeddings = model.embed(["tests/data/birds_10s.wav"])
    assert isinstance(embeddings, pd.DataFrame)
    assert len(embeddings) > 0


def test_rana_sierrae_cnn_train(model, train_df):
    """Test Rana Sierrae CNN training method."""
    model.freeze_feature_extractor()
    model.train(train_df, epochs=2)
