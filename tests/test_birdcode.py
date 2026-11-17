"""Tests for BirdCODE.subset_classes using a tiny randomly initialized BirdCODE-shaped network
(avoids downloading the real ~1.1GB checkpoint)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sound_event_detection")
pytest.importorskip("transformers")

import torch.nn as nn
from transformers import ConvNextConfig, ConvNextModel
from sound_event_detection.models import FrameDetector
from sound_event_detection.models.audioprotopnet_frame_detector import (
    AudioProtoPNetEncoder,
    _LinearLayerWithoutNegativeConnections,
)

from bioacoustics_model_zoo.birdcode import BirdCODE

LABELS = [f"Genus species{i}" for i in range(8)]
PROTOS_PER_CLASS = 4  # 20 in the real model
AUDIO = Path(__file__).parent / "data" / "birds_10s.wav"


def _tiny_network():
    torch.manual_seed(0)
    cfg = ConvNextConfig(
        num_channels=1, hidden_sizes=[8, 8, 16, 16], depths=[1, 1, 1, 1]
    )
    protos = nn.Parameter(
        torch.randn(len(LABELS) * PROTOS_PER_CLASS, cfg.hidden_sizes[-1], 1, 1)
    )
    enc = AudioProtoPNetEncoder(
        ConvNextModel(cfg),
        nn.Upsample(scale_factor=2, mode="bilinear"),
        protos,
        torch.device("cpu"),
    )
    head = _LinearLayerWithoutNegativeConnections(
        len(LABELS) * PROTOS_PER_CLASS, len(LABELS)
    )
    nn.init.uniform_(head.weight, 0.1, 1.0)
    nn.init.normal_(head.bias)
    net = FrameDetector(enc, list(LABELS))
    net.classifier = head
    return net.eval()


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(
        FrameDetector, "from_hf_hub", classmethod(lambda cls, *a, **k: _tiny_network())
    )
    return BirdCODE(device="cpu")


def test_subset_matches_full_model(model):
    full = model.predict([str(AUDIO)], batch_size=2, progress_bar=False)
    pick = [LABELS[5], LABELS[1], LABELS[7]]

    assert model.subset_classes(pick) is model
    sub = model.predict([str(AUDIO)], batch_size=2, progress_bar=False)

    assert list(sub.columns) == pick
    assert model.classes == pick
    assert model.network.labels == pick
    assert model.network.encoder.prototype_vectors.shape[0] == 3 * PROTOS_PER_CLASS
    assert model.network.classifier.out_features == 3
    assert model.embedding_dim == 3 * PROTOS_PER_CLASS
    np.testing.assert_allclose(sub.values, full[pick].values, rtol=1e-5, atol=1e-6)


def test_predict_frames_after_subset(model):
    full = model.predict_frames([str(AUDIO)], batch_size=2)
    pick = [LABELS[3], LABELS[0]]
    model.subset_classes(pick)
    sub = model.predict_frames([str(AUDIO)], batch_size=2)
    assert list(sub.columns) == pick
    np.testing.assert_allclose(sub.values, full[pick].values, rtol=1e-5, atol=1e-6)


def test_repeated_subsetting(model):
    full = model.predict([str(AUDIO)], progress_bar=False)
    model.subset_classes([LABELS[0], LABELS[2], LABELS[4]])
    model.subset_classes([LABELS[4], LABELS[0]])
    out = model.predict([str(AUDIO)], progress_bar=False)
    assert list(out.columns) == [LABELS[4], LABELS[0]]
    np.testing.assert_allclose(
        out.values, full[[LABELS[4], LABELS[0]]].values, rtol=1e-5, atol=1e-6
    )


def test_invalid_class_lists(model):
    with pytest.raises(ValueError, match="at least one"):
        model.subset_classes([])
    with pytest.raises(ValueError, match="duplicate"):
        model.subset_classes([LABELS[0], LABELS[0]])
    with pytest.raises(ValueError, match="not in self.classes"):
        model.subset_classes(["Not a bird"])
    # a failed call must leave the model untouched
    assert model.classes == LABELS
    assert (
        model.network.encoder.prototype_vectors.shape[0]
        == len(LABELS) * PROTOS_PER_CLASS
    )


def test_ambiguous_class_names_rejected(model):
    model.classes = ["a", "b", "a"] + LABELS[
        3:
    ]  # e.g. two species sharing a common name
    with pytest.raises(ValueError, match="more than one"):
        model.subset_classes(["a"])


def test_encoder_embeddings_computed_without_autograd(model):
    """opensoundscape's __call__ doesn't disable autograd, so batch_forward must"""
    grad_modes = []
    encoder = model.network.encoder
    original_forward = encoder.forward

    def spy(x):
        grad_modes.append(torch.is_grad_enabled())
        return original_forward(x)

    encoder.forward = spy

    class FakeSample:
        data = type("D", (), {"samples": np.zeros(160000, dtype=np.float32)})

    out = model.batch_forward([FakeSample()], targets=["encoder"], avgpool=False)
    assert grad_modes == [False]
    assert out["encoder"].shape[-1] == len(LABELS) * PROTOS_PER_CLASS
