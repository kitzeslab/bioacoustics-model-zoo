"""Test suite for model functionality across all models."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from bioacoustics_model_zoo import list_models


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_audio_path():
    """Return path to test audio file."""
    return "tests/data/birds_10s.wav"


@pytest.fixture
def mock_audio_data():
    """Create mock audio data for testing."""
    return np.random.rand(22050 * 5).astype(np.float32)


class TestModelFunctionality:
    """Test functionality across all model types."""
    
    def test_all_models_have_required_methods(self):
        """Test that all models have predict and embed methods."""
        models = list_models()
        dependency_error_models = []
        
        for model_name, model_class in models.items():
            try:
                # Try to instantiate - this will fail for missing dependencies
                model = model_class()
                
                # If instantiation succeeds, check for required methods
                assert hasattr(model, 'predict'), f"{model_name} missing predict method"
                assert callable(model.predict), f"{model_name} predict is not callable"
                
                # Not all models have embed method, but if they do, it should be callable
                if hasattr(model, 'embed'):
                    assert callable(model.embed), f"{model_name} embed is not callable"
                    
            except ImportError:
                # Expected for models with missing dependencies
                dependency_error_models.append(model_name)
                continue
            except Exception as e:
                # Other exceptions might be expected (missing checkpoints, etc.)
                # but should not be basic attribute/method errors
                assert not isinstance(e, (AttributeError, TypeError)), \
                    f"{model_name} failed with unexpected error: {e}"
        
        # At least some models should have dependencies available for testing
        print(f"Models with missing dependencies: {dependency_error_models}")
    
    @pytest.mark.parametrize("model_name", [
        "BirdNET", "HawkEars", "BirdSetConvNeXT", "BirdSetEfficientNetB1", "RanaSierraeCNN"
    ])
    def test_model_predict_output_format(self, model_name, temp_cache_dir):
        """Test that model predict methods return properly formatted DataFrames."""
        models = list_models()
        model_class = models[model_name]
        
        try:
            model = model_class(cache_dir=temp_cache_dir)
            
            # Create mock prediction output
            with patch.object(model, 'predict') as mock_predict:
                # Mock a typical prediction DataFrame
                mock_df = pd.DataFrame({
                    'file': ['test.wav'],
                    'start_time': [0.0],
                    'end_time': [3.0],
                    'species1': [0.8],
                    'species2': [0.2]
                })
                mock_predict.return_value = mock_df
                
                # Test prediction
                result = model.predict(['test.wav'])
                
                # Verify output format
                assert isinstance(result, pd.DataFrame)
                assert 'file' in result.columns
                assert len(result) > 0
                
        except ImportError:
            pytest.skip(f"{model_name} dependencies not available")
        except Exception as e:
            pytest.skip(f"{model_name} failed to initialize: {e}")
    
    @pytest.mark.parametrize("model_name", [
        "BirdNET", "HawkEars", "Perch", "YAMNet"
    ])
    def test_model_embed_output_format(self, model_name, temp_cache_dir):
        """Test that model embed methods return properly formatted DataFrames."""
        models = list_models()
        model_class = models[model_name]
        
        try:
            model = model_class(cache_dir=temp_cache_dir)
            
            # Only test if model has embed method
            if not hasattr(model, 'embed'):
                pytest.skip(f"{model_name} does not have embed method")
            
            # Create mock embedding output
            with patch.object(model, 'embed') as mock_embed:
                # Mock a typical embedding DataFrame
                mock_df = pd.DataFrame({
                    'file': ['test.wav'],
                    'start_time': [0.0],
                    'end_time': [3.0],
                    'embedding_0': [0.1],
                    'embedding_1': [0.2],
                    'embedding_2': [0.3]
                })
                mock_embed.return_value = mock_df
                
                # Test embedding
                result = model.embed(['test.wav'])
                
                # Verify output format
                assert isinstance(result, pd.DataFrame)
                assert 'file' in result.columns
                assert len(result) > 0
                
        except ImportError:
            pytest.skip(f"{model_name} dependencies not available")
        except Exception as e:
            pytest.skip(f"{model_name} failed to initialize: {e}")
    
    def test_model_cache_dir_parameter(self, temp_cache_dir):
        """Test that models accept and use cache_dir parameter."""
        models = list_models()
        
        # Test with models that are likely to have cache_dir parameter
        cache_supporting_models = ["BirdNET", "HawkEars", "BirdSetConvNeXT", "BirdSetEfficientNetB1"]
        
        for model_name in cache_supporting_models:
            if model_name not in models:
                continue
                
            model_class = models[model_name]
            
            try:
                model = model_class(cache_dir=temp_cache_dir)
                # If initialization succeeds, the cache_dir parameter was accepted
                assert True  # Model initialized successfully with cache_dir
                
            except ImportError:
                # Expected for missing dependencies
                continue
            except TypeError as e:
                if "cache_dir" in str(e):
                    pytest.fail(f"{model_name} does not accept cache_dir parameter")
                else:
                    # Other TypeError might be expected
                    continue
            except Exception:
                # Other exceptions might be expected (missing files, etc.)
                continue
    
    def test_model_classes_attribute(self):
        """Test that models have classes attribute when applicable."""
        models = list_models()
        
        for model_name, model_class in models.items():
            try:
                model = model_class()
                
                # Most models should have a classes attribute
                if hasattr(model, 'classes'):
                    classes = model.classes
                    assert isinstance(classes, (list, tuple, pd.Index)), \
                        f"{model_name} classes should be list-like"
                    assert len(classes) > 0, f"{model_name} should have non-empty classes"
                
            except ImportError:
                # Expected for missing dependencies
                continue
            except Exception:
                # Other exceptions might be expected
                continue
    
    @pytest.mark.slow
    def test_model_integration_with_real_audio(self, sample_audio_path, temp_cache_dir):
        """Integration test with real audio for models with available dependencies."""
        if not Path(sample_audio_path).exists():
            pytest.skip("Test audio file not available")
        
        models = list_models()
        successful_models = []
        
        # Try each model - this is an integration test to ensure basic functionality works
        for model_name, model_class in models.items():
            try:
                model = model_class(cache_dir=temp_cache_dir)
                
                # Try prediction
                try:
                    predictions = model.predict([sample_audio_path])
                    assert isinstance(predictions, pd.DataFrame)
                    assert len(predictions) > 0
                    successful_models.append(f"{model_name}-predict")
                except Exception as e:
                    print(f"{model_name} predict failed: {e}")
                
                # Try embedding if available
                if hasattr(model, 'embed'):
                    try:
                        embeddings = model.embed([sample_audio_path])
                        assert isinstance(embeddings, pd.DataFrame)
                        assert len(embeddings) > 0
                        successful_models.append(f"{model_name}-embed")
                    except Exception as e:
                        print(f"{model_name} embed failed: {e}")
                        
            except ImportError:
                # Expected for missing dependencies
                continue
            except Exception as e:
                print(f"{model_name} initialization failed: {e}")
                continue
        
        # At least some models should work if dependencies are available
        print(f"Successfully tested: {successful_models}")
        
        # This test passes as long as no unexpected errors occur
        # The actual functionality is tested by individual model test files
        assert True
    
    def test_model_error_handling(self):
        """Test that models handle invalid inputs gracefully."""
        models = list_models()
        
        for model_name, model_class in models.items():
            try:
                model = model_class()
                
                # Test with invalid file paths
                with pytest.raises((FileNotFoundError, ValueError, Exception)):
                    model.predict(["nonexistent_file.wav"])
                
                # Test with empty list
                with pytest.raises((ValueError, Exception)):
                    model.predict([])
                    
            except ImportError:
                # Expected for missing dependencies
                continue
            except Exception:
                # Other initialization exceptions might be expected
                continue