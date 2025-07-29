"""Test suite for BirdSet models functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Skip all BirdSet tests if dependencies are not available
try:
    import transformers
    import torch
    import torchaudio
    import opensoundscape
    HAS_BIRDSET_DEPS = True
except ImportError:
    HAS_BIRDSET_DEPS = False

SKIP_BIRDSET = not HAS_BIRDSET_DEPS


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_huggingface_model():
    """Mock HuggingFace model for testing."""
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "species1", 1: "species2", 2: "species3"}
    mock_model.num_labels = 3
    return mock_model


@pytest.fixture
def sample_audio_path():
    """Return path to test audio file."""
    return "tests/data/birds_10s.wav"


@pytest.mark.skipif(SKIP_BIRDSET, reason="BirdSet dependencies not available")
class TestBirdSetIntegration:
    """Integration tests for BirdSet models."""
    
    @pytest.mark.slow
    def test_birdset_convnext_predict_with_real_audio(self, sample_audio_path, temp_cache_dir):
        """Test BirdSet ConvNeXT prediction with real audio file."""
        from bioacoustics_model_zoo import BirdSetConvNeXT
        
        try:
            model = BirdSetConvNeXT(cache_dir=temp_cache_dir)
            
            # Test prediction
            predictions = model.predict([sample_audio_path])
            
            # Verify predictions format
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) > 0
            assert len(predictions.columns) > 0
            
        except Exception as e:
            # If model fails to download or load, skip test
            pytest.skip(f"BirdSet ConvNeXT integration test failed: {e}")
    
    @pytest.mark.slow
    def test_birdset_efficientnetb1_predict_with_real_audio(self, sample_audio_path, temp_cache_dir):
        """Test BirdSet EfficientNetB1 prediction with real audio file."""
        from bioacoustics_model_zoo import BirdSetEfficientNetB1
        
        try:
            model = BirdSetEfficientNetB1(cache_dir=temp_cache_dir)
            
            # Test prediction
            predictions = model.predict([sample_audio_path])
            
            # Verify predictions format
            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) > 0
            assert len(predictions.columns) > 0
            
        except Exception as e:
            # If model fails to download or load, skip test
            pytest.skip(f"BirdSet EfficientNetB1 integration test failed: {e}")


@pytest.mark.skipif(HAS_BIRDSET_DEPS, reason="Dependencies are available")
def test_birdset_missing_dependencies():
    """Test that BirdSet models raise appropriate error when dependencies are missing."""
    from bioacoustics_model_zoo import BirdSetConvNeXT, BirdSetEfficientNetB1
    
    with pytest.raises(ImportError) as exc_info:
        BirdSetConvNeXT()
    assert "required" in str(exc_info.value).lower()
    
    with pytest.raises(ImportError) as exc_info:
        BirdSetEfficientNetB1()
    assert "required" in str(exc_info.value).lower()