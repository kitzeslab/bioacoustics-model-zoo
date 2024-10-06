import pytest
import sys

import pandas as pd

from bioacoustics_model_zoo.rana_sierrae_cnn import rana_sierrae_cnn


@pytest.fixture
def model():
    return rana_sierrae_cnn()


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
    pass


def test_rana_sierrae_cnn_predict(model):
    model.predict(["tests/data/birds_10s.wav"], batch_size=2)
    pass


def test_rana_sierrae_cnn_embed(model):
    model.embed(["tests/data/birds_10s.wav"])
    pass


def test_rana_sierrae_cnn_train(model, train_df):
    model.freeze_feature_extractor()
    model.train(train_df, epochs=2)
