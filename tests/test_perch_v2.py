"""Test suite for Perch2 model functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# Skip all Perch2 tests if dependencies are not available
try:
    import tensorflow as tf  # noqa: F401
    import opensoundscape  # noqa: F401
    from bioacoustics_model_zoo import Perch2

    HAS_PERCH2_DEPS = True
except ImportError:
    HAS_PERCH2_DEPS = False

SKIP_PERCH2 = not HAS_PERCH2_DEPS


@pytest.fixture
def fake_perch2_hub_assets(tmp_path):
    """Create fake TF Hub assets expected by Perch2.__init__."""
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Required class list files read during initialization
    (assets_dir / "labels.csv").write_text("label\nsp1\nsp2\n")
    (assets_dir / "perch_v2_ebird_classes.csv").write_text("ebird_code\nsp1\nsp2\n")

    return tmp_path


@pytest.mark.skipif(SKIP_PERCH2, reason="Perch2 dependencies not available")
class TestPerch2:
    """Test Perch2 model functionality."""

    @patch("tensorflow_hub.resolve")
    @patch("tensorflow_hub.load")
    def test_perch2_initialization(
        self, mock_hub_load, mock_hub_resolve, fake_perch2_hub_assets
    ):
        """Test Perch2 initialization with mocked TF Hub model and local assets."""
        mock_tf_model = MagicMock()
        mock_hub_load.return_value = mock_tf_model
        mock_hub_resolve.return_value = str(fake_perch2_hub_assets)

        model = Perch2(device="cpu", version=1)

        expected_path = (
            "https://www.kaggle.com/models/google/bird-vocalization-classifier/"
            "tensorFlow2/perch_v2_cpu/1"
        )
        mock_hub_load.assert_called_once_with(expected_path)
        mock_hub_resolve.assert_called_once_with(expected_path)

        assert model.tf_model is mock_tf_model
        assert model.device == "cpu"
        assert model.version == 1
        assert hasattr(model, "classes")
        assert len(model.classes) == 2
        assert hasattr(model, "preprocessor")

    @patch("tensorflow_hub.resolve")
    @patch("tensorflow_hub.load")
    def test_perch2_core_methods_exist(
        self, mock_hub_load, mock_hub_resolve, fake_perch2_hub_assets
    ):
        """Test that Perch2 exposes core prediction/embedding interfaces."""
        mock_hub_load.return_value = MagicMock()
        mock_hub_resolve.return_value = str(fake_perch2_hub_assets)

        model = Perch2(device="cpu", version=1)

        assert hasattr(model, "predict")
        assert callable(model.predict)
        assert hasattr(model, "embed")
        assert callable(model.embed)
        assert hasattr(model, "forward")
        assert callable(model.forward)


@pytest.mark.skipif(HAS_PERCH2_DEPS, reason="Dependencies are available")
def test_perch2_missing_dependencies():
    """Test that Perch2 raises an ImportError when TensorFlow deps are unavailable."""
    from bioacoustics_model_zoo import Perch2

    with pytest.raises(ImportError) as exc_info:
        Perch2()

    assert "required" in str(exc_info.value).lower()
