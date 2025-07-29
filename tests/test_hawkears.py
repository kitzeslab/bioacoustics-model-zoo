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
    from bioacoustics_model_zoo.hawkears.hawkears import HawkEars
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
            "train_class_codes": ["C1", "C2", "C3"]
        },
        "state_dict": {}
    }


@pytest.fixture
def model():
    if not HAS_HAWKEARS_DEPS:
        pytest.skip("HawkEars dependencies not available")
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


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
def test_hawkears_init(model):
    """Test that HawkEars initializes correctly."""
    assert hasattr(model, 'classes')
    assert hasattr(model, 'preprocessor')
    assert hasattr(model, 'network')
    assert len(model.classes) > 0


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
def test_hawkears_init_no_weights():
    """Test HawkEars initialization without downloading weights."""
    # This test needs to be updated since checkpoint_url parameter doesn't exist
    pass


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
@pytest.mark.skipif(not Path("tests/data/birds_10s.wav").exists(), 
                   reason="Test audio file not available")
def test_hawkears_predict(model):
    """Test HawkEars prediction method."""
    predictions = model.predict(["tests/data/birds_10s.wav"], batch_size=2)
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) > 0


@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
@pytest.mark.skipif(not Path("tests/data/birds_10s.wav").exists(),
                   reason="Test audio file not available")
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


# Cache-related tests
@pytest.mark.skipif(SKIP_HAWKEARS, reason="HawkEars dependencies not available")
class TestHawkEarsCache:
    """Test HawkEars caching functionality."""
    
    @patch('bioacoustics_model_zoo.utils.download_cached_file')
    @patch('torch.load')
    @patch('bioacoustics_model_zoo.hawkears.architecture_constructors.get_hgnet')
    def test_hawkears_custom_cache_dir(self, mock_get_hgnet, mock_torch_load, mock_download,
                                     temp_cache_dir, mock_checkpoint_data):
        """Test HawkEars with custom cache directory."""
        checkpoint_files = []
        for i in range(1, 6):
            ckpt_path = Path(temp_cache_dir) / f"hgnet{i}.ckpt"
            ckpt_path.touch()
            checkpoint_files.append(ckpt_path)
        
        mock_download.side_effect = checkpoint_files
        mock_torch_load.return_value = mock_checkpoint_data
        mock_get_hgnet.return_value = MagicMock()
        
        # Initialize with custom cache_dir
        model = HawkEars(cache_dir=temp_cache_dir)
        
        # Verify cache_dir was passed to download calls
        for call in mock_download.call_args_list:
            args, kwargs = call
            assert kwargs['cache_dir'] == temp_cache_dir
            assert kwargs['model_name'] == 'hawkears'
    
    @patch('bioacoustics_model_zoo.utils.download_cached_file')
    @patch('torch.load')
    @patch('bioacoustics_model_zoo.hawkears.architecture_constructors.get_hgnet')
    def test_hawkears_force_reload(self, mock_get_hgnet, mock_torch_load, mock_download,
                                 temp_cache_dir, mock_checkpoint_data):
        """Test HawkEars with force_reload option."""
        checkpoint_files = []
        for i in range(1, 6):
            ckpt_path = Path(temp_cache_dir) / f"hgnet{i}.ckpt"
            ckpt_path.touch()
            checkpoint_files.append(ckpt_path)
        
        mock_download.side_effect = checkpoint_files
        mock_torch_load.return_value = mock_checkpoint_data
        mock_get_hgnet.return_value = MagicMock()
        
        # Initialize with force_reload=True
        model = HawkEars(force_reload=True, cache_dir=temp_cache_dir)
        
        # Verify redownload_existing was set to True
        for call in mock_download.call_args_list:
            args, kwargs = call
            assert kwargs['redownload_existing'] is True


@pytest.mark.skipif(HAS_HAWKEARS_DEPS, reason="Dependencies are available")
def test_hawkears_missing_dependencies():
    """Test that HawkEars raises appropriate error when dependencies are missing."""
    from bioacoustics_model_zoo import HawkEars
    
    with pytest.raises(ImportError) as exc_info:
        HawkEars()
    
    assert "required" in str(exc_info.value).lower()
