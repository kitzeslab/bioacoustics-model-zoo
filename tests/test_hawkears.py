import pytest
import sys

import pandas as pd

from bioacoustics_model_zoo.hawkears.hawkears import HawkEars


@pytest.fixture
def model():
    return HawkEars()


@pytest.fixture
def train_df():
    df = pd.DataFrame(
        {
            "file": ["tests/data/birds_10s.wav"] * 3,
            "start_time": [0, 3, 6],
            "end_time": [3, 6, 9],
            "Black-and-white Warbler": [1, 0, 1],
        }
    ).set_index(["file", "start_time", "end_time"])

    return df


def test_hawkears_init(model):
    pass


def test_hawkears_init_no_weights():
    model = HawkEars(checkpoint_url=None)


def test_hawkears_predict(model):
    model.predict(["tests/data/birds_10s.wav"], batch_size=2)
    pass


def test_hawkears_embed(model):
    model.embed(["tests/data/birds_10s.wav"])
    pass


def test_hawkears_train(model, train_df):
    model.change_classes(train_df.columns)
    model.freeze_feature_extractor()
    model.train(train_df, epochs=2)
