"""
Test suite for cache module functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from bioacoustics_model_zoo import cache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = Path(__file__).parent / "temp_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_file():
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("test content")
        temp_file = f.name

    yield temp_file

    # Cleanup after test
    Path(temp_file).unlink(missing_ok=True)


class TestCacheModule:
    """Test cache module functionality."""

    def test_get_default_cache_dir(self):
        """Test getting default cache directory."""
        default_dir = cache.get_default_cache_dir()
        assert isinstance(default_dir, str)
        assert len(default_dir) > 0
        assert "bioacoustics_model_zoo" in default_dir

    def test_set_default_cache_dir(self, temp_cache_dir):
        """Test setting custom default cache directory."""
        original_dir = cache.get_default_cache_dir()

        # Set custom cache dir
        cache.set_default_cache_dir(temp_cache_dir)
        assert cache.get_default_cache_dir() == str(temp_cache_dir)

        # Reset to original
        cache.set_default_cache_dir(original_dir)
        assert cache.get_default_cache_dir() == original_dir

    def test_get_model_cache_dir(self, temp_cache_dir):
        """Test getting model-specific cache directory."""
        model_cache = cache.get_model_cache_dir("test_model", cache_dir=temp_cache_dir)

        assert isinstance(model_cache, Path)
        assert model_cache.exists()
        assert model_cache.name == "test_model"
        assert str(model_cache.parent) == str(temp_cache_dir)

    def test_get_model_cache_dir_default(self):
        """Test getting model cache dir with default cache."""
        model_cache = cache.get_model_cache_dir("test_model")

        assert isinstance(model_cache, Path)
        assert "test_model" in str(model_cache)

    def test_is_cached_false(self, temp_cache_dir):
        """Test checking for non-existent cached file."""
        assert not cache.is_cached("nonexistent.txt", "test_model", cache_dir=temp_cache_dir)

    def test_is_cached_true(self, temp_cache_dir, test_file):
        """Test checking for existing cached file."""
        # Copy test file to cache
        model_cache_dir = cache.get_model_cache_dir("test_model", cache_dir=temp_cache_dir)
        cached_file = model_cache_dir / "test.txt"
        shutil.copy2(test_file, cached_file)

        assert cache.is_cached("test.txt", "test_model", cache_dir=temp_cache_dir)

    def test_get_cached_file_path(self, temp_cache_dir):
        """Test getting path to cached file."""
        file_path = cache.get_cached_file_path("test.txt", "test_model", cache_dir=temp_cache_dir)

        assert isinstance(file_path, Path)
        assert file_path.name == "test.txt"
        assert "test_model" in str(file_path)

    def test_save_to_cache(self, temp_cache_dir, test_file):
        """Test saving file to cache."""
        cached_path = cache.save_to_cache(
            test_file, "saved_test.txt", "test_model", cache_dir=temp_cache_dir
        )

        assert isinstance(cached_path, Path)
        assert cached_path.exists()
        assert cached_path.name == "saved_test.txt"
        assert "test_model" in str(cached_path)

        # Verify file content
        with open(cached_path, "r") as f:
            content = f.read()
        assert content == "test content"

    def test_save_to_cache_no_overwrite(self, temp_cache_dir, test_file):
        """Test that save_to_cache doesn't overwrite existing files."""
        # Save file once
        cached_path1 = cache.save_to_cache(
            test_file, "no_overwrite.txt", "test_model", temp_cache_dir
        )

        # Modify the original file
        with open(test_file, "w") as f:
            f.write("modified content")

        # Try to save again - should not overwrite
        cached_path2 = cache.save_to_cache(
            test_file, "no_overwrite.txt", "test_model", temp_cache_dir
        )

        assert cached_path1 == cached_path2

        # Verify original content is preserved
        with open(cached_path2, "r") as f:
            content = f.read()
        assert content == "test content"
    
    def test_clear_cached_model(self, temp_cache_dir, test_file):
        """Test clearing cached files for a specific model."""
        # Save some files to cache
        cache.save_to_cache(test_file, "file1.txt", "test_model", cache_dir=temp_cache_dir)
        cache.save_to_cache(test_file, "file2.txt", "test_model", cache_dir=temp_cache_dir)
        cache.save_to_cache(test_file, "file3.txt", "other_model", cache_dir=temp_cache_dir)
        
        # Verify files exist
        assert cache.is_cached("file1.txt", "test_model", cache_dir=temp_cache_dir)
        assert cache.is_cached("file2.txt", "test_model", cache_dir=temp_cache_dir)
        assert cache.is_cached("file3.txt", "other_model", cache_dir=temp_cache_dir)
        
        # Clear specific model cache
        cache.clear_cached_model("test_model", cache_dir=temp_cache_dir)
        
        # Verify test_model files are gone but other_model files remain
        assert not cache.is_cached("file1.txt", "test_model", cache_dir=temp_cache_dir)
        assert not cache.is_cached("file2.txt", "test_model", cache_dir=temp_cache_dir)
        assert cache.is_cached("file3.txt", "other_model", cache_dir=temp_cache_dir)
    
    def test_clear_all_cached_models(self, temp_cache_dir, test_file):
        """Test clearing all cached models."""
        # Save files for multiple models
        cache.save_to_cache(test_file, "file1.txt", "model1", cache_dir=temp_cache_dir)
        cache.save_to_cache(test_file, "file2.txt", "model2", cache_dir=temp_cache_dir)
        cache.save_to_cache(test_file, "file3.txt", "model3", cache_dir=temp_cache_dir)
        
        # Verify files exist
        assert cache.is_cached("file1.txt", "model1", cache_dir=temp_cache_dir)
        assert cache.is_cached("file2.txt", "model2", cache_dir=temp_cache_dir)
        assert cache.is_cached("file3.txt", "model3", cache_dir=temp_cache_dir)
        
        # Clear all caches
        cache.clear_all_cached_models(cache_dir=temp_cache_dir)
        
        # Verify all files are gone
        assert not cache.is_cached("file1.txt", "model1", cache_dir=temp_cache_dir)
        assert not cache.is_cached("file2.txt", "model2", cache_dir=temp_cache_dir)
        assert not cache.is_cached("file3.txt", "model3", cache_dir=temp_cache_dir)
    
    def test_cache_with_model_version(self, temp_cache_dir, test_file):
        """Test caching with model version parameter."""
        # Save files with different versions
        path_v1 = cache.get_cached_file_path("model.txt", "test_model", model_version="v1.0", cache_dir=temp_cache_dir)
        path_v2 = cache.get_cached_file_path("model.txt", "test_model", model_version="v2.0", cache_dir=temp_cache_dir)
        
        # Paths should be different for different versions
        assert path_v1 != path_v2
        assert "v1.0" in str(path_v1)
        assert "v2.0" in str(path_v2)
        
        # Save same filename with different versions
        cached_v1 = cache.save_to_cache(test_file, "model.txt", "test_model", model_version="v1.0", cache_dir=temp_cache_dir)
        cached_v2 = cache.save_to_cache(test_file, "model.txt", "test_model", model_version="v2.0", cache_dir=temp_cache_dir)
        
        # Both should exist independently
        assert cached_v1.exists()
        assert cached_v2.exists()
        assert cached_v1 != cached_v2
        
        # Check that both are cached
        assert cache.is_cached("model.txt", "test_model", model_version="v1.0", cache_dir=temp_cache_dir)
        assert cache.is_cached("model.txt", "test_model", model_version="v2.0", cache_dir=temp_cache_dir)
    
    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directories are created as needed."""
        # Get cache dir for non-existent model
        model_cache = cache.get_model_cache_dir("new_model", cache_dir=temp_cache_dir)
        
        # Directory should be created
        assert model_cache.exists()
        assert model_cache.is_dir()
        assert model_cache.name == "new_model"
        
        # Test with version
        versioned_cache = cache.get_model_cache_dir("new_model", model_version="v1.0", cache_dir=temp_cache_dir)
        assert versioned_cache.exists()
        assert versioned_cache.is_dir()
        assert "v1.0" in str(versioned_cache)
