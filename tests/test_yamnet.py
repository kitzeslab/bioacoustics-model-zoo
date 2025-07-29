"""Test suite for YAMNet model functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Skip all YAMNet tests if dependencies are not available
try:
    import tensorflow as tf
    import opensoundscape
    from bioacoustics_model_zoo import YAMNet
    HAS_YAMNET_DEPS = True
except ImportError:
    HAS_YAMNET_DEPS = False

SKIP_YAMNET = not HAS_YAMNET_DEPS


@pytest.mark.skipif(SKIP_YAMNET, reason="YAMNet dependencies not available")
class TestYAMNet:
    """Test YAMNet model functionality."""
    
    @patch('tensorflow_hub.load')
    def test_yamnet_initialization(self, mock_hub_load):
        """Test YAMNet model initialization."""
        mock_model = MagicMock()
        mock_hub_load.return_value = mock_model
        
        # Initialize model
        model = YAMNet()
        
        # Verify model has expected attributes
        assert hasattr(model, 'tf_model')
        assert hasattr(model, 'classes')
        assert hasattr(model, 'preprocessor')
    
    @patch('tensorflow_hub.load')
    def test_yamnet_predict_method_exists(self, mock_hub_load):
        """Test that YAMNet has predict method."""
        mock_hub_load.return_value = MagicMock()
        
        model = YAMNet()
        
        # Verify predict method exists
        assert hasattr(model, 'predict')
        assert callable(model.predict)
    
    @patch('tensorflow_hub.load')
    def test_yamnet_embed_method_exists(self, mock_hub_load):
        """Test that YAMNet has embed method."""
        mock_hub_load.return_value = MagicMock()
        
        model = YAMNet()
        
        # Verify embed method exists
        assert hasattr(model, 'embed')
        assert callable(model.embed)


@pytest.mark.skipif(HAS_YAMNET_DEPS, reason="Dependencies are available")
def test_yamnet_missing_dependencies():
    """Test that YAMNet raises appropriate error when dependencies are missing."""
    from bioacoustics_model_zoo import YAMNet
    
    with pytest.raises(ImportError) as exc_info:
        YAMNet()
    
    assert "required" in str(exc_info.value).lower()