"""Test suite for conditional import system functionality."""

import pytest
from bioacoustics_model_zoo.utils import register_bmz_model


class TestConditionalImports:
    """Test the conditional import system for optional dependencies."""
    
    def test_missing_dependency_class_structure(self):
        """Test that missing dependency classes have proper structure."""
        from bioacoustics_model_zoo import MissingTFDependency, MissingTFLiteDependency
        
        # Test error message structure
        with pytest.raises(ImportError) as exc_info:
            MissingTFDependency()
        assert "Tensorflow" in str(exc_info.value)
        assert "required" in str(exc_info.value)
        
        with pytest.raises(ImportError) as exc_info:
            MissingTFLiteDependency()
        assert "ai_edge_litert" in str(exc_info.value)
        assert "required" in str(exc_info.value)
    
    def test_conditional_model_imports(self):
        """Test that models are conditionally imported based on dependencies."""
        from bioacoustics_model_zoo import list_models
        
        models = list_models()
        
        # All models should be present in some form (either real or missing dependency)
        expected_models = [
            "BirdNET", "Perch", "YAMNet", "SeparationModel",
            "HawkEars", "HawkEars_Embedding", "HawkEars_Low_Band", "HawkEars_v010",
            "BirdSetConvNeXT", "BirdSetEfficientNetB1", "RanaSierraeCNN"
        ]
        
        for model_name in expected_models:
            assert model_name in models, f"Model {model_name} not found in list_models()"
            assert models[model_name] is not None
    
    def test_birdnet_dependency_handling(self):
        """Test BirdNET handles dependencies correctly."""
        from bioacoustics_model_zoo import list_models
        
        models = list_models()
        assert "BirdNET" in models
        
        # Try to instantiate - it should either work or raise ImportError
        # this test currently fails with Mac if ai-edge-litert is > 1.3
        # and ai-edge-litert is no longer provided for < 2.0 on mac
        # see https://github.com/google-ai-edge/LiteRT/issues/2836 and 
        # https://github.com/google-ai-edge/LiteRT/issues/4385
        BirdNETClass = models["BirdNET"]
        try:
            model = BirdNETClass()
            assert hasattr(model, 'predict')  # Should have predict method if successful
        except ImportError as e:
            assert "required" in str(e).lower() or "missing" in str(e).lower()
    
    def test_model_registration_with_missing_deps(self):
        """Test that models with missing dependencies are still registered."""
        from bioacoustics_model_zoo.utils import BMZ_MODEL_LIST
        
        # Create a mock missing dependency class
        @register_bmz_model
        class MockMissingDependency:
            """Mock model with missing dependencies."""
            def __init__(self):
                raise ImportError("Mock dependency missing")
        
        # Should be registered despite raising ImportError
        assert MockMissingDependency in BMZ_MODEL_LIST
        
        # Should appear in list_models
        from bioacoustics_model_zoo import list_models
        models = list_models()
        assert "MockMissingDependency" in models
    
    def test_hawkears_dependency_combinations(self):
        """Test HawkEars dependency handling for different missing combinations."""
        # This tests the logic: if timm is None or torchaudio is None
        
        # We can't easily mock the import-time checks, but we can test 
        # that the MissingHawkearsDependency class works correctly
        from bioacoustics_model_zoo import MissingHawkearsDependency
        
        with pytest.raises(ImportError) as exc_info:
            MissingHawkearsDependency()
        
        error_msg = str(exc_info.value).lower()
        assert "timm" in error_msg
        assert "torchaudio" in error_msg
        assert "required" in error_msg
    
    def test_birdset_dependency_combinations(self):
        """Test BirdSet dependency handling."""
        from bioacoustics_model_zoo import MissingBirdSetDependency
        
        with pytest.raises(ImportError) as exc_info:
            MissingBirdSetDependency()
        
        error_msg = str(exc_info.value)
        assert "BirdSetConvNeXT" in error_msg
        assert "transformers" in error_msg
    
    def test_all_models_have_proper_error_handling(self):
        """Test that all models either work or raise appropriate ImportError."""
        from bioacoustics_model_zoo import list_models
        
        models = list_models()
        
        for model_name, model_class in models.items():
            try:
                # Try to instantiate the model
                instance = model_class()
                # If it works, it should have basic attributes
                # (This test will only pass for models with available dependencies)
                
            except ImportError as e:
                # If it raises ImportError, that's expected for missing dependencies
                assert "required" in str(e).lower() or "missing" in str(e).lower()
                
            except Exception as e:
                # Other exceptions might be okay (e.g., file not found for checkpoints)
                # but we should not get syntax errors or attribute errors
                assert not isinstance(e, (SyntaxError, AttributeError))
    
    def test_import_fallback_behavior(self):
        """Test that the import system gracefully handles missing packages."""
        # Test the pattern used in __init__.py for handling missing imports
        
        # Simulate the try/except pattern
        try:
            import nonexistent_package
            has_package = True
        except ImportError:
            has_package = False
            nonexistent_package = None
        
        assert not has_package
        assert nonexistent_package is None
        
        # This pattern should allow the conditional import logic to work
        if nonexistent_package is None:
            class MockMissingClass:
                def __init__(self):
                    raise ImportError("Package not available")
            
            assert MockMissingClass is not None
            with pytest.raises(ImportError):
                MockMissingClass()