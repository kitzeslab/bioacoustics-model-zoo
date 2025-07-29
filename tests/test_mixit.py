"""Test suite for MixIT separation model functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Skip all MixIT tests if dependencies are not available
try:
    import tensorflow as tf
    import opensoundscape
    from bioacoustics_model_zoo import SeparationModel
    HAS_MIXIT_DEPS = True
except ImportError:
    HAS_MIXIT_DEPS = False

SKIP_MIXIT = not HAS_MIXIT_DEPS


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_checkpoint_path(temp_cache_dir):
    """Create a mock checkpoint file."""
    checkpoint_dir = Path(temp_cache_dir) / "model_checkpoint"
    checkpoint_dir.mkdir(parents=True)
    
    # Create mock checkpoint files
    for filename in ["model.ckpt-123.data-00000-of-00001", "model.ckpt-123.index"]:
        (checkpoint_dir / filename).touch()
    
    return str(checkpoint_dir / "model.ckpt-123")


@pytest.fixture
def sample_audio_path():
    """Return path to test audio file."""
    return "tests/data/birds_10s.wav"


@pytest.fixture
def mock_audio_data():
    """Create mock audio data for testing."""
    # Create mock audio samples (22050 Hz * 5 seconds)
    samples = np.random.rand(22050 * 5).astype(np.float32)
    return samples


@pytest.mark.skipif(SKIP_MIXIT, reason="MixIT dependencies not available")
class TestSeparationModel:
    """Test MixIT SeparationModel functionality."""
    
    @patch('tensorflow.saved_model.load')
    def test_separation_model_initialization(self, mock_tf_load, mock_checkpoint_path):
        """Test SeparationModel initialization."""
        # Mock TensorFlow saved model
        mock_model = MagicMock()
        mock_tf_load.return_value = mock_model
        
        # Initialize model
        model = SeparationModel(checkpoint=mock_checkpoint_path)
        
        # Verify model has expected attributes
        assert hasattr(model, 'model')
        assert hasattr(model, 'sample_rate')
        assert model.sample_rate == 22050
    
    @patch('tensorflow.saved_model.load')
    def test_separation_model_separate_waveform_method(self, mock_tf_load, mock_checkpoint_path, mock_audio_data):
        """Test separate_waveform method exists and callable."""
        mock_model = MagicMock()
        
        # Mock the model's prediction method
        num_sources = 4
        mock_output = np.random.rand(1, len(mock_audio_data), num_sources).astype(np.float32)
        mock_model.signatures = {'waveform_to_separated_audio': MagicMock(return_value={'separated_audio': mock_output})}
        mock_tf_load.return_value = mock_model
        
        model = SeparationModel(checkpoint=mock_checkpoint_path)
        
        # Test separate_waveform method exists
        assert hasattr(model, 'separate_waveform')
        assert callable(model.separate_waveform)
    
    @patch('tensorflow.saved_model.load')
    def test_separation_model_separate_audio_method(self, mock_tf_load, mock_checkpoint_path):
        """Test separate_audio method exists and callable."""
        mock_model = MagicMock()
        mock_tf_load.return_value = mock_model
        
        model = SeparationModel(checkpoint=mock_checkpoint_path)
        
        # Test separate_audio method exists
        assert hasattr(model, 'separate_audio')
        assert callable(model.separate_audio)
    
    @patch('tensorflow.saved_model.load')
    def test_separation_model_load_separate_write_method(self, mock_tf_load, mock_checkpoint_path):
        """Test load_separate_write method exists and callable."""
        mock_model = MagicMock()
        mock_tf_load.return_value = mock_model
        
        model = SeparationModel(checkpoint=mock_checkpoint_path)
        
        # Test load_separate_write method exists
        assert hasattr(model, 'load_separate_write')
        assert callable(model.load_separate_write)


@pytest.mark.skipif(HAS_MIXIT_DEPS, reason="Dependencies are available")
def test_separation_model_missing_dependencies():
    """Test that SeparationModel raises appropriate error when dependencies are missing."""
    from bioacoustics_model_zoo import SeparationModel
    
    with pytest.raises(ImportError) as exc_info:
        SeparationModel(checkpoint="dummy_path")
    
    assert "required" in str(exc_info.value).lower()