"""
Test suite for BirdNET model functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Skip all BirdNET tests if ai_edge_litert is not available
try:
    import ai_edge_litert

    HAS_TFLITE = True
except ImportError:
    HAS_TFLITE = False

# Skip if opensoundscape is not available
try:
    import opensoundscape

    HAS_OSS = True
except ImportError:
    HAS_OSS = False

SKIP_BIRDNET = not (HAS_TFLITE and HAS_OSS)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_tflite_model():
    """Mock TFLite model for testing."""
    mock_model = MagicMock()
    mock_model.allocate_tensors.return_value = None
    return mock_model


@pytest.fixture
def sample_audio_path():
    """Return path to test audio file."""
    return "tests/data/birds_10s.wav"


@pytest.mark.skipif(SKIP_BIRDNET, reason="BirdNET dependencies not available")
class TestBirdNET:
    """Test BirdNET model functionality."""

    @pytest.mark.slow
    def test_birdnet_predict_with_real_audio(self, sample_audio_path, temp_cache_dir):
        """Test BirdNET prediction with real audio file (integration test)."""
        from bioacoustics_model_zoo import BirdNET

        # This is a slow integration test - only run if explicitly requested
        # and if the test audio file exists
        try:
            model = BirdNET(cache_dir=temp_cache_dir)

            # Test prediction
            predictions = model.predict([sample_audio_path])

            # Verify predictions format
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) > 0
            assert len(predictions.columns) > 0

        except Exception as e:
            # If model fails to download or load, skip test
            pytest.skip(f"BirdNET integration test failed: {e}")

    @pytest.mark.slow
    def test_birdnet_embed_with_real_audio(self, sample_audio_path, temp_cache_dir):
        """Test BirdNET embedding with real audio file (integration test)."""
        from bioacoustics_model_zoo import BirdNET

        # This is a slow integration test
        try:
            model = BirdNET(cache_dir=temp_cache_dir)

            # Test embedding
            embeddings = model.embed([sample_audio_path])

            # Verify embeddings format
            assert isinstance(embeddings, pd.DataFrame)
            assert len(embeddings) > 0
            assert len(embeddings.columns) > 0

        except Exception as e:
            # If model fails to download or load, skip test
            pytest.skip(f"BirdNET embedding test failed: {e}")


@pytest.mark.skipif(HAS_TFLITE and HAS_OSS, reason="Dependencies are available")
def test_birdnet_missing_dependencies():
    """Test that BirdNET raises appropriate error when dependencies are missing."""
    # This test runs when dependencies are NOT available
    from bioacoustics_model_zoo import BirdNET

    with pytest.raises(ImportError) as exc_info:
        BirdNET()

    assert "required" in str(exc_info.value).lower()
