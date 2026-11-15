import warnings

import numpy as np
from opensoundscape.utils import identity
import pandas as pd

from bioacoustics_model_zoo.utils import (
    AudioSampleArrayDataloader,
    download_cached_file,
    register_bmz_model,
    verify_file_sha256,
)
from bioacoustics_model_zoo.tensorflow_wrapper import (
    TensorFlowModelWithPytorchClassifier,
)

# VGGish front end with the settings HOWLish was trained on
# (detection_pipeline/vggish_params.py in github.com/CamposRF/HOWLish)
SAMPLE_RATE = 8000
WINDOW = 400  # 0.05 s STFT window
HOP = 80  # 0.01 s
FFT_LENGTH = 512
N_FRAMES = 96  # one 0.96 s example
LOG_OFFSET = 0.01
EXAMPLE_HOP = 0.96  # examples do not overlap
# an example's 96 frames read 0.96 s + one trailing window of audio
SAMPLE_DURATION = ((N_FRAMES - 1) * HOP + WINDOW) / SAMPLE_RATE  # 1.0 s


def _mel_matrix(n_mels=64, low_hz=125.0, high_hz=2000.0):
    """VGGish's (HTK-style) mel filterbank, shape (FFT_LENGTH // 2 + 1, n_mels)"""

    def to_mel(hz):
        return 1127.0 * np.log(1.0 + hz / 700.0)

    bins_mel = to_mel(np.linspace(0, SAMPLE_RATE / 2, FFT_LENGTH // 2 + 1))
    edges = np.linspace(to_mel(low_hz), to_mel(high_hz), n_mels + 2)
    lower, center, upper = edges[:-2], edges[1:-1], edges[2:]
    weights = np.maximum(
        0,
        np.minimum(
            (bins_mel[:, None] - lower) / (center - lower),
            (upper - bins_mel[:, None]) / (upper - center),
        ),
    )
    weights[0, :] = 0  # VGGish drops the DC bin
    return weights


MEL_MATRIX = _mel_matrix()
HANN = 0.5 - 0.5 * np.cos(2 * np.pi / WINDOW * np.arange(WINDOW))  # periodic


def n_examples(n_samples):
    """whole 0.96 s examples in n_samples of 8 kHz audio, counted as VGGish does"""
    frames = 1 + (n_samples - WINDOW) // HOP
    return max(0, 1 + (frames - N_FRAMES) // N_FRAMES)


def log_mel_examples(clips):
    """HOWLish input for a batch of 1.0 s, 8 kHz clips: shape (batch, 96, 64, 1)

    Log-mel spectrogram as in VGGish, then each example scaled to [-1, 1],
    as in HOWLish_pipeline.py.
    """
    idx = np.arange(N_FRAMES)[:, None] * HOP + np.arange(WINDOW)
    frames = clips[:, idx] * HANN  # (batch, 96, 400)
    spec = np.abs(np.fft.rfft(frames, FFT_LENGTH))
    x = np.log(spec @ MEL_MATRIX + LOG_OFFSET)
    lo = x.min(axis=(1, 2), keepdims=True)
    span = x.max(axis=(1, 2), keepdims=True) - lo
    # an all-constant example (digital silence) would divide by zero in the original
    x = 2 * (x - lo) / np.where(span > 0, span, 1) - 1
    return x[..., None].astype(np.float32)


@register_bmz_model
class HOWLish(TensorFlowModelWithPytorchClassifier):
    """HOWLish wolf howl detector (Campos et al. 2025)

    A VGGish CNN fine-tuned on 50,137 hours of soundscapes from the Iberian Peninsula
    to detect wolf (Canis lupus) howls in 8 kHz audio. It scores 0.96 s examples
    from 0 (no howl) to 1 (howl).

    Campos, R., Krofel, M., Rio-Maior, H. and Renna, F. (2025) HOWLish: a CNN for
    automated wolf howl detection. Remote Sensing in Ecology and Conservation 12(1).
    https://doi.org/10.1002/rse2.70024
    Code and pipeline: https://github.com/CamposRF/HOWLish
    Weights: https://zenodo.org/records/24201346 (GPL-3.0-or-later)

    .predict() follows the authors' detection pipeline v1.0.0: scores on back-to-back
    0.96 s examples, averaged over a moving window of `window_size` examples (W = 3
    in the paper). Each output row is one such window, 0.96 s apart. The authors keep
    windows above 0.90 (T) and review 110 s of audio around each one. Those review
    steps (threshold, clips, dropping overlapping clips) are not part of .predict(),
    see detection_pipeline/HOWLish_pipeline.py in the authors' repository.
    Scores match the authors' for 8 kHz recordings. Other sample rates are resampled
    per 1 s example rather than per file, so scores can differ slightly.
    Inference only: embed() and train() are not supported.

    Args:
        checkpoint_path: local path to the frozen graph (HOWLish_fg_100.pb);
            default downloads it to the model zoo cache

    Example:
    ```
    import bioacoustics_model_zoo as bmz
    model = bmz.HOWLish()
    scores = model.predict(['soundscape.wav'])  # smoothed, W = 3
    raw = model.predict(['soundscape.wav'], window_size=1)  # per 0.96 s example
    candidates = scores[scores['wolf_howl'] > 0.90]
    ```
    """

    url = "https://zenodo.org/records/24201346/files/HOWLish_fg_100.pb"
    sha256 = "583ec9e1fa464842d5e14de7c1fb0428a2e5a917134186027b42d03ab26f1965"

    def __init__(self, checkpoint_path=None):
        import tensorflow as tf

        if checkpoint_path is None:
            checkpoint_path = download_cached_file(
                self.url,
                filename="HOWLish_fg_100.pb",
                model_name="howlish",
                model_version="1.0.0",
                sha256=self.sha256,
            )
            # download_cached_file does not check the hash after a fresh download
            if not verify_file_sha256(checkpoint_path, self.sha256):
                raise ValueError(
                    f"{checkpoint_path} does not match the expected SHA256; "
                    "delete it and try again"
                )

        graph_def = tf.compat.v1.GraphDef()
        with open(checkpoint_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        wrapped = tf.compat.v1.wrap_function(
            lambda: tf.compat.v1.import_graph_def(graph_def, name=""), []
        )
        tf_model = wrapped.prune(
            wrapped.graph.as_graph_element("x:0"),
            wrapped.graph.as_graph_element("Identity:0"),
        )

        super().__init__(
            embedding_size=1,
            classes=["wolf_howl"],
            sample_duration=SAMPLE_DURATION,
            sample_rate=SAMPLE_RATE,
        )
        self.tf_model = tf_model
        self.name = "HOWLish"
        self.version = "1.0.0"
        self.inference_dataloader_cls = AudioSampleArrayDataloader

    def embed(self, *args, **kwargs):
        raise NotImplementedError(
            "HOWLish is inference only: the graph exposes only the score"
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "HOWLish is inference only: the graph exposes only the score"
        )

    def batch_forward(self, batch_samples, targets=(-1,), avgpool=False):
        """scores (batch, 1) for a batch of AudioSamples of 1.0 s"""
        import tensorflow as tf

        clips = np.array([s.data.samples for s in batch_samples], dtype=np.float64)
        x = tf.convert_to_tensor(log_mel_examples(clips))
        return {-1: self.tf_model(x).numpy()}

    def predict(self, samples, window_size=3, **kwargs):
        """wolf howl scores, averaged over `window_size` consecutive 0.96 s examples

        Args:
            samples: audio file path or list of paths (whole recordings)
            window_size: examples per moving average, 3 in the authors' pipeline;
                1 returns the raw per-example scores
            **kwargs: passed to the dataloader, e.g. batch_size, num_workers.
                clip_overlap / clip_step are fixed by the model and ignored.

        Returns:
            DataFrame with index (file, start_time, end_time) and column 'wolf_howl'
        """
        import librosa

        for key in ("clip_overlap", "clip_overlap_fraction", "clip_step"):
            if kwargs.pop(key, None):
                warnings.warn(f"HOWLish ignores {key}: examples are 0.96 s apart")
        files = [samples] if isinstance(samples, str) else list(samples)

        # Back-to-back examples as in VGGish framing; a partial one at the end is
        # dropped. Starts get 1 us added because librosa truncates offset * rate,
        # and e.g. 67 * 0.96 * 8000 = 514559.99..., one sample early.
        clips, counts = [], []
        for file in files:
            try:
                duration = librosa.get_duration(path=file)
            except Exception as exc:
                warnings.warn(f"HOWLish skips {file}: {exc}")
                duration = 0
            n = n_examples(round(duration * SAMPLE_RATE))
            counts.append(n)
            clips += [(file, k * EXAMPLE_HOP + 1e-6) for k in range(n)]
        if not clips:
            return pd.DataFrame(
                columns=self.classes,
                index=pd.MultiIndex.from_tuples(
                    [], names=["file", "start_time", "end_time"]
                ),
            )
        clip_df = pd.DataFrame(
            [(f, t, t + SAMPLE_DURATION) for f, t in clips],
            columns=["file", "start_time", "end_time"],
        ).set_index(["file", "start_time", "end_time"])
        dl = self.predict_dataloader(clip_df, collate_fn=identity, **kwargs)
        scores = []
        for batch in dl:
            s = self.batch_forward(batch)[-1][:, 0].astype(float)
            # clips that failed to load come back as zero placeholders, as in CNN.predict
            s[[sample.is_alternative for sample in batch]] = np.nan
            scores.append(s)
        scores = np.concatenate(scores)
        dl.dataset.report()  # warns about clips that failed to load, as CNN.predict

        rows = []
        per_file = np.split(scores, np.cumsum(counts)[:-1])
        for file, file_scores in zip(files, per_file):
            if len(file_scores) < window_size:
                continue
            avg = np.convolve(file_scores, np.ones(window_size) / window_size, "valid")
            for k, score in enumerate(avg):
                # same span as the authors' pipeline: window_size x 0.96 s
                start = round(k * EXAMPLE_HOP, 2)
                end = round((k + window_size) * EXAMPLE_HOP, 2)
                rows.append((file, start, end, score))
        df = pd.DataFrame(rows, columns=["file", "start_time", "end_time", "wolf_howl"])
        return df.set_index(["file", "start_time", "end_time"])
