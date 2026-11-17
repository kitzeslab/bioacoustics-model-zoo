import os

import numpy as np
import pytest
import soundfile as sf

import bioacoustics_model_zoo  # loads torch before tensorflow, the other order aborts

tf = pytest.importorskip("tensorflow")

from bioacoustics_model_zoo import howlish
from bioacoustics_model_zoo.howlish import HOWLish, log_mel_examples, n_examples


@pytest.fixture
def tiny_graph(tmp_path):
    """stand-in frozen graph with HOWLish's input and output names"""
    g = tf.Graph()
    with g.as_default():
        x = tf.compat.v1.placeholder(tf.float32, [None, 96, 64, 1], name="x")
        tf.identity(tf.reduce_mean(x, axis=[1, 2]), name="Identity")
    path = tmp_path / "tiny.pb"
    path.write_bytes(g.as_graph_def().SerializeToString())
    return str(path)


@pytest.fixture
def index_wav(tmp_path):
    """20 s at 8 kHz where each sample's value encodes its own position"""
    n = 20 * 8000
    path = tmp_path / "index.wav"
    sf.write(path, np.arange(n) / n, 8000, subtype="FLOAT")
    return str(path), n


def test_log_mel_examples():
    rng = np.random.default_rng(0)
    x = log_mel_examples(rng.normal(size=(2, 8000)) * 0.1)
    assert x.shape == (2, 96, 64, 1)
    assert np.allclose(x.min(axis=(1, 2, 3)), -1)
    assert np.allclose(x.max(axis=(1, 2, 3)), 1)
    assert np.isfinite(log_mel_examples(np.zeros((1, 8000)))).all()


def test_predict_runs_graph(tiny_graph, index_wav):
    path, _ = index_wav
    df = HOWLish(checkpoint_path=tiny_graph).predict(path, batch_size=4)
    assert list(df.columns) == ["wolf_howl"]
    assert len(df) == 18  # 20 whole examples in 20 s, minus 2 for the average


def test_examples_are_back_to_back(tiny_graph, index_wav):
    """each example starts exactly k * 7680 samples in, also where k * 0.96 * 8000
    lands just below the integer in floating point (k = 11 and 15 here)"""
    path, n = index_wav
    model = HOWLish(checkpoint_path=tiny_graph)
    model.batch_forward = lambda batch, **kw: {
        -1: np.array([[s.data.samples[0] * n] for s in batch])
    }
    raw = model.predict([path], window_size=1, batch_size=4)
    k = np.arange(len(raw))
    assert np.allclose(raw["wolf_howl"].to_numpy(), k * 7680, atol=0.01)
    assert np.allclose(raw.index.get_level_values("start_time"), k * 0.96)


def test_moving_average(tiny_graph, index_wav):
    path, n = index_wav
    model = HOWLish(checkpoint_path=tiny_graph)
    model.batch_forward = lambda batch, **kw: {
        -1: np.array([[round(s.data.samples[0] * n / 7680)] for s in batch])
    }
    with pytest.warns(UserWarning, match="clip_overlap"):
        df = model.predict([path], clip_overlap=0.5)  # Dipper passes clip_overlap
    assert np.allclose(df["wolf_howl"].to_numpy(), np.arange(len(df)) + 1)
    start = df.index.get_level_values("start_time").to_numpy()
    end = df.index.get_level_values("end_time").to_numpy()
    assert np.allclose(end - start, 3 * 0.96)


def test_example_count_in_samples(tiny_graph, tmp_path):
    """307,520 samples hold 40 examples; counting in float seconds gave 39"""
    assert [n_examples(n) for n in (7999, 8000, 307519, 307520)] == [0, 1, 39, 40]
    path = tmp_path / "edge.wav"
    sf.write(path, np.zeros(307520), 8000)
    raw = HOWLish(checkpoint_path=tiny_graph).predict([str(path)], window_size=1)
    assert len(raw) == 40


def test_files_stay_separate(tiny_graph, index_wav, tmp_path):
    path, _ = index_wav
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    with pytest.warns(UserWarning, match="skips"):
        df = HOWLish(checkpoint_path=tiny_graph).predict([path, str(bad), path])
    assert len(df) == 2 * 18  # the same file twice is not averaged across the join


def test_rejects_wrong_download(tiny_graph, monkeypatch):
    monkeypatch.setattr(howlish, "download_cached_file", lambda *a, **k: tiny_graph)
    with pytest.raises(ValueError, match="SHA256"):
        HOWLish()


def test_short_file(tiny_graph, tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(4000), 8000)
    df = HOWLish(checkpoint_path=tiny_graph).predict([str(path)])
    assert len(df) == 0 and list(df.columns) == ["wolf_howl"]


@pytest.mark.slow
@pytest.mark.skipif(
    "HOWLISH_CHECKPOINT" not in os.environ, reason="set HOWLISH_CHECKPOINT to the .pb"
)
def test_real_weights():
    model = HOWLish(checkpoint_path=os.environ["HOWLISH_CHECKPOINT"])
    df = model.predict(["tests/data/birds_10s.wav"])
    assert len(df) > 0
    assert df["wolf_howl"].between(0, 1).all()
