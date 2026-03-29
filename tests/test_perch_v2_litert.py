"""Test suite for Perch2LiteRT model functionality."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Skip Perch2LiteRT functional tests if opensoundscape/model module deps are unavailable
try:
    import opensoundscape  # noqa: F401
    from bioacoustics_model_zoo.perch_v2_litert import Perch2LiteRT

    HAS_LITERT_TEST_DEPS = True
except ImportError:
    HAS_LITERT_TEST_DEPS = False

try:
    import ai_edge_litert  # noqa: F401

    HAS_REAL_LITERT = True
except ImportError:
    HAS_REAL_LITERT = False

SKIP_LITERT = not HAS_LITERT_TEST_DEPS


@pytest.fixture
def fake_model_file(tmp_path):
    """Create a placeholder .tflite file path for initialization tests."""
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"fake tflite content")
    return model_path


@pytest.fixture
def fake_litert_modules():
    """Provide mocked ai_edge_litert modules and interpreter behavior."""
    runner = MagicMock(
        return_value={
            "embedding": np.zeros((1, 1536), dtype=np.float32),
            "spatial_embedding": np.zeros((1, 5, 3, 1536), dtype=np.float32),
            "label": np.zeros((1, 14795), dtype=np.float32),
            "spectrogram": np.zeros((1, 64, 10), dtype=np.float32),
        }
    )

    mock_interpreter_instance = MagicMock()
    mock_interpreter_instance.allocate_tensors.return_value = None
    mock_interpreter_instance.get_input_details.return_value = [
        {"dtype": np.float32, "shape": np.array([1, 160000], dtype=np.int32)}
    ]
    mock_interpreter_instance.get_signature_list.return_value = {
        "serving_default": {"inputs": ["inputs"]}
    }
    mock_interpreter_instance.get_signature_runner.return_value = runner

    mock_interpreter_cls = MagicMock(return_value=mock_interpreter_instance)

    ai_edge_litert_mod = types.ModuleType("ai_edge_litert")
    interpreter_mod = types.ModuleType("ai_edge_litert.interpreter")
    interpreter_mod.Interpreter = mock_interpreter_cls
    ai_edge_litert_mod.interpreter = interpreter_mod

    return (
        {
            "ai_edge_litert": ai_edge_litert_mod,
            "ai_edge_litert.interpreter": interpreter_mod,
        },
        mock_interpreter_cls,
        runner,
    )


@pytest.mark.skipif(SKIP_LITERT, reason="Perch2LiteRT test dependencies not available")
class TestPerch2LiteRT:
    """Test Perch2LiteRT model functionality."""

    def test_litert_initialization(self, fake_model_file, fake_litert_modules):
        """Test Perch2LiteRT initialization with mocked LiteRT interpreter."""
        fake_modules, mock_interpreter_cls, _ = fake_litert_modules

        with patch.dict(sys.modules, fake_modules):
            model = Perch2LiteRT(model_path=str(fake_model_file), num_tflite_threads=2)

        mock_interpreter_cls.assert_called_once_with(
            model_path=str(Path(fake_model_file).resolve()),
            num_threads=2,
        )
        assert hasattr(model, "tf_model")
        assert hasattr(model, "preprocessor")
        assert hasattr(model, "predict")
        assert callable(model.predict)

    def test_litert_batch_forward_selected_outputs(
        self, fake_model_file, fake_litert_modules
    ):
        """Test that batch_forward returns only requested targets and supports -1 key."""
        fake_modules, _, _ = fake_litert_modules

        with patch.dict(sys.modules, fake_modules):
            model = Perch2LiteRT(model_path=str(fake_model_file))
            batch = np.zeros((1, 160000), dtype=np.float32)
            outputs = model.batch_forward(batch, targets=("label", "embedding", -1))

        assert "label" in outputs
        assert "embedding" in outputs
        assert -1 in outputs
        assert outputs["label"].shape[0] == 1
        assert outputs["embedding"].shape[1] == 1536


@pytest.mark.skipif(HAS_REAL_LITERT, reason="ai_edge_litert is available")
def test_litert_missing_dependency_error(fake_model_file):
    """Test that Perch2LiteRT raises ModuleNotFoundError without ai_edge_litert."""
    from bioacoustics_model_zoo.perch_v2_litert import Perch2LiteRT

    with pytest.raises(ModuleNotFoundError) as exc_info:
        Perch2LiteRT(model_path=str(fake_model_file))

    assert "ai-edge-litert" in str(exc_info.value).lower()
