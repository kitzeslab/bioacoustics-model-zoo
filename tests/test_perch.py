"""Test suite for Perch model functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Skip all Perch tests if dependencies are not available
try:
    import tensorflow as tf
    import opensoundscape
    from bioacoustics_model_zoo import Perch
    HAS_PERCH_DEPS = True
except ImportError:
    HAS_PERCH_DEPS = False

SKIP_PERCH = not HAS_PERCH_DEPS


@pytest.fixture
def sample_audio_path():
    """Return path to test audio file."""
    return "tests/data/birds_10s.wav"


@pytest.mark.skipif(SKIP_PERCH, reason="Perch dependencies not available")
class TestPerch:
    """Test Perch model functionality."""
    
    @patch('tensorflow_hub.load')
    def test_perch_initialization(self, mock_hub_load):
        """Test Perch model initialization."""
        mock_model = MagicMock()
        mock_hub_load.return_value = mock_model
        
        # Initialize model
        model = Perch()
        
        # Verify TensorFlow Hub model was loaded
        mock_hub_load.assert_called_once()
        
        # Verify model has expected attributes
        assert hasattr(model, 'tf_model')
        assert hasattr(model, 'classes')
        assert hasattr(model, 'preprocessor')
    
    @patch('tensorflow_hub.load')
    def test_perch_predict_method_exists(self, mock_hub_load):
        """Test that Perch has predict method."""
        mock_hub_load.return_value = MagicMock()
        
        model = Perch()
        
        # Verify predict method exists
        assert hasattr(model, 'predict')
        assert callable(model.predict)
    
    @patch('tensorflow_hub.load')
    def test_perch_embed_method_exists(self, mock_hub_load):
        """Test that Perch has embed method."""
        mock_hub_load.return_value = MagicMock()
        
        model = Perch()
        
        # Verify embed method exists
        assert hasattr(model, 'embed')
        assert callable(model.embed)


@pytest.mark.skipif(HAS_PERCH_DEPS, reason="Dependencies are available")
def test_perch_missing_dependencies():
    """Test that Perch raises appropriate error when dependencies are missing."""
    from bioacoustics_model_zoo import Perch
    
    with pytest.raises(ImportError) as exc_info:
        Perch()
    
    assert "required" in str(exc_info.value).lower()