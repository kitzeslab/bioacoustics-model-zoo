import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

# Skip all HawkEars tests if dependencies are not available
try:
    import timm
    import torchaudio
    import torch
    import opensoundscape
    from bioacoustics_model_zoo.hawkears.hawkears import (
        HawkEars,
        HawkEars_Embedding,
        HawkEars_Low_Band,
    )

    HAS_HAWKEARS_DEPS = True
except ImportError:
    HAS_HAWKEARS_DEPS = False

SKIP_HAWKEARS = not HAS_HAWKEARS_DEPS


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_checkpoint_data():
    """Mock checkpoint data for HawkEars."""
    return {
        "hyper_parameters": {
            "model_name": "hgnet_tiny",
            "train_class_names": ["Class1", "Class2", "Class3"],
            "train_class_codes": ["C1", "C2", "C3"],
        },
        "state_dict": {},
    }


@pytest.fixture
def model():
    if not HAS_HAWKEARS_DEPS:
        pytest.skip("HawkEars dependencies not available")
    return HawkEars()


@pytest.fixture
def low_band_model():
    if not HAS_HAWKEARS_DEPS:
        pytest.skip("HawkEars dependencies not available")
    return HawkEars_Low_Band()


@pytest.fixture
def embedding_model():
    if not HAS_HAWKEARS_DEPS:
        pytest.skip("HawkEars dependencies not available")
    return HawkEars_Embedding()


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


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
def test_hawkears_init(model):
    """Test that HawkEars initializes correctly."""
    assert hasattr(model, "classes")
    assert hasattr(model, "preprocessor")
    assert hasattr(model, "network")
    assert len(model.classes) > 0


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
def test_hawkears_init_no_weights():
    """Test HawkEars initialization without downloading weights."""
    # This test needs to be updated since checkpoint_url parameter doesn't exist
    pass


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
@pytest.mark.skipif(
    not Path("tests/data/birds_10s.wav").exists(),
    reason="Test audio file not available",
)
def test_hawkears_predict(model):
    """Test HawkEars prediction method."""
    predictions = model.predict(["tests/data/birds_10s.wav"], batch_size=2)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) > 0


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
@pytest.mark.skipif(
    not Path("tests/data/birds_10s.wav").exists(),
    reason="Test audio file not available",
)
def test_hawkears_low_band_predict(low_band_model):
    """Test HawkEars prediction method."""
    predictions = low_band_model.predict(["tests/data/birds_10s.wav"], batch_size=2)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) > 0


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
@pytest.mark.skipif(
    not Path("tests/data/birds_10s.wav").exists(),
    reason="Test audio file not available",
)
def test_hawkears_embedding_predict(embedding_model):
    """Test HawkEars prediction method."""
    predictions = embedding_model.predict(["tests/data/birds_10s.wav"], batch_size=2)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) > 0


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
@pytest.mark.skipif(
    not Path("tests/data/birds_10s.wav").exists(),
    reason="Test audio file not available",
)
def test_hawkears_embed(model):
    """Test HawkEars embedding method."""
    embeddings = model.embed(["tests/data/birds_10s.wav"])
    assert isinstance(embeddings, pd.DataFrame)
    assert len(embeddings) > 0


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
def test_hawkears_train(model, train_df):
    """Test HawkEars training method."""
    model.change_classes(train_df.columns)
    model.freeze_feature_extractor()
    model.train(train_df, epochs=2)


@pytest.mark.skipif(HAS_HAWKEARS_DEPS, reason="Dependencies are available")
def test_hawkears_missing_dependencies():
    """Test that HawkEars raises appropriate error when dependencies are missing."""
    from bioacoustics_model_zoo import HawkEars

    with pytest.raises(ImportError) as exc_info:
        HawkEars()

    assert "required" in str(exc_info.value).lower()
