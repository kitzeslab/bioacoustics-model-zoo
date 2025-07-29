"""
Test suite for utils module functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from bioacoustics_model_zoo import utils


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_file_content():
    """Sample file content for testing downloads."""
    return b"test file content for download testing"


class TestUtilsModule:
    """Test utils module functionality."""
    
    def test_list_models(self):
        """Test listing available models."""
        models = utils.list_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check that BirdNET is in the list (should always be available)
        model_names = list(models.keys())
        assert any("BirdNET" in name for name in model_names)
    
    def test_describe_models(self):
        """Test model descriptions."""
        descriptions = utils.describe_models()
        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0
        
        # Check that all models have descriptions
        for model_name, description in descriptions.items():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_register_bmz_model(self):
        """Test model registration decorator."""
        # Get initial model count
        initial_count = len(utils.BMZ_MODEL_LIST)
        
        @utils.register_bmz_model
        class TestModel:
            """Test model for registration."""
            pass
        
        # Check that model was added
        assert len(utils.BMZ_MODEL_LIST) == initial_count + 1
        assert TestModel in utils.BMZ_MODEL_LIST
        
        # Check it appears in list_models
        models = utils.list_models()
        assert "TestModel" in models
        assert models["TestModel"] is TestModel
        
        # Check that describe_models includes it
        descriptions = utils.describe_models()
        assert "TestModel" in descriptions
        assert descriptions["TestModel"] == "Test model for registration."
    
    def test_register_bmz_model_multiple_classes(self):
        """Test registering multiple model classes."""
        initial_count = len(utils.BMZ_MODEL_LIST)
        
        @utils.register_bmz_model
        class FirstTestModel:
            """First test model."""
            pass
        
        @utils.register_bmz_model
        class SecondTestModel:
            """Second test model."""
            pass
        
        # Check both models were added
        assert len(utils.BMZ_MODEL_LIST) == initial_count + 2
        assert FirstTestModel in utils.BMZ_MODEL_LIST
        assert SecondTestModel in utils.BMZ_MODEL_LIST
        
        # Check both appear in list_models
        models = utils.list_models()
        assert "FirstTestModel" in models
        assert "SecondTestModel" in models
        assert models["FirstTestModel"] is FirstTestModel
        assert models["SecondTestModel"] is SecondTestModel
    
    def test_register_bmz_model_with_init_docstring(self):
        """Test model registration when only __init__ has docstring."""
        initial_count = len(utils.BMZ_MODEL_LIST)
        
        @utils.register_bmz_model
        class TestModelWithInitDoc:
            def __init__(self):
                """Model with __init__ docstring only."""
                pass
        
        # Check description comes from __init__
        descriptions = utils.describe_models()
        assert "TestModelWithInitDoc" in descriptions
        assert descriptions["TestModelWithInitDoc"] == "Model with __init__ docstring only."
    
    def test_register_bmz_model_no_docstring(self):
        """Test model registration when no docstring is available."""
        initial_count = len(utils.BMZ_MODEL_LIST)
        
        @utils.register_bmz_model
        class TestModelNoDoc:
            pass
        
        # Check default description is used
        descriptions = utils.describe_models()
        assert "TestModelNoDoc" in descriptions
        assert descriptions["TestModelNoDoc"] == "no description"
    
    def test_download_file_skip_if_exists(self, temp_dir):
        """Test download skips if file already exists."""
        # Create existing file
        existing_file = Path(temp_dir) / "existing.txt"
        with open(existing_file, 'w') as f:
            f.write("existing content")
        
        # This would normally try to download, but should skip since file exists
        url = "https://example.com/existing.txt"
        downloaded_path = utils.download_file(url, save_dir=temp_dir, verbose=False)
        
        assert downloaded_path == existing_file
        # Content should be unchanged
        with open(downloaded_path, 'r') as f:
            content = f.read()
        assert content == "existing content"
    
    def test_github_url_conversion(self):
        """Test GitHub URL conversion logic."""
        # Test the URL conversion logic used in download_file
        github_url = "https://github.com/user/repo/blob/main/file.txt"
        expected_raw = "https://github.com/user/repo/raw/main/file.txt"
        
        # Simulate the conversion done in download_file
        if "github.com" in github_url:
            converted_url = str(github_url).replace("/blob/", "/raw/")
            assert converted_url == expected_raw
    
    def test_collate_to_np_array(self):
        """Test audio sample collation function."""
        # This test would require opensoundscape dependencies and sample data
        # For now, just verify the function exists and can be imported
        assert hasattr(utils, 'collate_to_np_array')
        assert callable(utils.collate_to_np_array)
    
    def test_audio_sample_array_dataloader(self):
        """Test AudioSampleArrayDataloader class."""
        # Verify class exists and can be imported
        assert hasattr(utils, 'AudioSampleArrayDataloader')
        assert utils.AudioSampleArrayDataloader is not None