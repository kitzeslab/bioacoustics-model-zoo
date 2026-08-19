"""Run HawkEars v2.x model ensembles in Sound Event Detection mode using BriteKit"""

from typing import Literal
import zipfile
import numpy as np

from pathlib import Path
from opensoundscape.ml.cnn import register_model_cls

from bioacoustics_model_zoo.utils import (
    download_cached_file,
    register_bmz_model,
)
from bioacoustics_model_zoo.hawkears.hawkears_base_config import Hawkears2Config
import yaml
import pandas as pd

CHECKPOINT_LOOKUP = {
    "2.2.0": {
        "main_models_url": (
            "https://github.com/jhuus/HawkEars/releases/download/models-2.2.0/main-models-2.2.0.zip"
        ),
        "main_models_sha256": (
            "24cf831c2f32affced1c7a87f8e92c5d4c5aff0ea2f8fe1e935502eb9c335c3b"
        ),
        "low_band_models_url": (
            "https://github.com/jhuus/HawkEars/releases/download/models-2.0.0/low-band-models-2.0.0.zip"
        ),
        "low_band_models_sha256": (
            "4d56c860b1f3a317cfbe3d10bd7ee98574dcf840e29d5b19bd4487e6ff418c55"
        ),
    }
}


# BriteKit's custom get_range function

try:
    from britekit.core.util import get_range
except ImportError:
    get_range = None


# HakwEars's custom spacing out of ensembled models' event times
# note that the logic seems to assume 3s clips even though clip_duration is configurable
def get_start_time_offsets(n_models, clip_duration):
    end_offset = clip_duration - 0.5
    initial_start_times = get_range(0, end_offset, 0.5)

    if n_models > 10:
        pass  # use the default initial_start_times from above
    elif n_models == 10:
        # space out the last 3
        initial_start_times.extend([0, 1, 2])
    elif n_models == 9:
        # space out the last 3
        initial_start_times.extend([0, 1, 2])
    elif n_models == 8:
        # space out the last 2
        initial_start_times.extend([0, 1.5])
    elif n_models == 4:
        initial_start_times = [
            0,
            0.75,
            1.5,
            2.25,
        ]
    elif n_models == 3:
        initial_start_times = [0, 1, 2]
    elif n_models == 2:
        initial_start_times = [0, 1.5]
    elif n_models == 1:
        initial_start_times = [0]

    # Remove any that go past end of clip
    for i in range(1, len(initial_start_times), 1):
        if initial_start_times[i] > end_offset:
            initial_start_times = initial_start_times[:i]
            break

    return initial_start_times


@register_bmz_model
@register_model_cls
class HawkEars2:
    """inference-only HawkEars v2.x ensembled models"""

    def __init__(self, cfg_path=None, version="2.2.0", device=None):
        """initialize HawkEars2 model ensemble for inference

        Note: requires britekit package (`pip install britekit`)

        Args:
            cfg_path: optionally override default config settings by providing alternative yaml file
            version: Version of the model checkpoints to load, currently only "2.2.0" is supported
            device: torch.device or str, defaults to gpu if available else cpu
                - e.g. "cuda:1", "mps", "cpu"

        # TODO: add support for 'lowband' model (replace class scores for 2 sp with that model's scores?)

        # always uses all models when doing frame-level SED?
        #https://github.com/jhuus/BriteKit/blob/f2ed8d39bfb96380ab981c794404812980a1c147/src/britekit/core/predictor.py#L292-L293

        """
        assert (
            version in CHECKPOINT_LOOKUP
        ), f"Version {version} not found in CHECKPOINT_LOOKUP. Available versions: {list(CHECKPOINT_LOOKUP.keys())}"

        try:
            import britekit
        except Exception as e:
            raise ImportError(
                f"britekit is a required dependency for HawkEars2. Please use `pip install britekit` to install it."
            ) from e

        # TODO: separate one for low-band model(s)
        archive_path = download_cached_file(
            CHECKPOINT_LOOKUP[version]["main_models_url"],
            filename=f"hawkears2_main_models_{version}.zip",
            model_name="HawkEars2",
            model_version=version,
            sha256=CHECKPOINT_LOOKUP[version]["main_models_sha256"],
        )

        # unzip archive if not already unzipped
        unzip_path = Path(archive_path).parent / Path(archive_path).stem
        if not unzip_path.exists():
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(Path(archive_path).parent)

        # load config yaml
        if cfg_path is None:
            cfg_path = Path(__file__).parent / f"hawkears-{version}_default.yaml"
        cfg = britekit.get_config(cfg_path)

        # directly use britekit predictor class since it does quite a few things
        # differently than opensoundscape's CNN class, including SED and handling ensembles where
        # each model predicts on distinct temporal windows
        self.predictor = britekit.Predictor(unzip_path, device=device, cfg=cfg)

        self.ensemble_offsets = get_start_time_offsets(
            len(self.predictor.models), self.predictor.cfg.audio.spec_duration
        )

        # default to common names for output labels
        self.predictor.cfg.infer.label_field = "name"

    @property
    def classes(self):
        """return list of class names from self.predictor.class_names

        (note there are also .codes, .alt_codes, and .alt_names in self.predictor)
        """
        return self.predictor.class_names

    @property
    def frame_len(self):
        return 1 / self.predictor.cfg.train.sed_fps

    def set_class_name_type(
        self, name_type: Literal["common", "alpha", "scientific", "eBird"]
    ):
        """Select common/scientific/alpha/eBird as output class names

        Outputs such as clip predictions and species detection labels will use the selected
        naming convention for species names. The default is "common" (common names).

        Args:
            name_type: str, one of "common", "alpha", "scientific",
            "eBird"
        """
        map = {
            "common": "name",
            "alpha": "code",
            "scientific": "alt_name",
            "eBird": "alt_code",
        }
        self.predictor.cfg.infer.label_field = map[name_type]

    def predict(self, audio_paths):
        """generate species prediction scores on fixed-length audio clips

        Args:
            audio_paths: str or Path to audio file, or list of str/Path to audio files

        Returns:
            pd.DataFrame with one row per (3-second) clip and a multi-index of
            file, start_time, end_time and columns for each class with
            prediction scores
        """
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]
        aggregated_scores = []
        for f in audio_paths:
            # run all ensembled models on the same clips, getting clip-level
            # scores averaged across the ensemble
            results, _, starts = self.predictor.get_recording_scores(f)
            start_times = np.array(starts)
            end_times = start_times + self.predictor.cfg.audio.spec_duration

            # place scores in a dataframe with multi-index of file, start_time, end_time
            df = pd.DataFrame(results, columns=self.predictor.class_names)
            df.index = pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"file": f, "start_time": start_times, "end_time": end_times}
                )
            )
            aggregated_scores.append(df)
        return pd.concat(aggregated_scores)

    def predict_frames(self, audio_paths):
        """Generate frame-level species prediction scores


        Args:
            audio_paths: str or Path to audio file, or list of str/Path to audio files

        Returns:
            pd.DataFrame with one row per (0.25-second) frame and a multi-index of
            file, start_time, end_time and columns for each class with
            prediction scores
        """
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]
        aggregated_scores = []
        for f in audio_paths:
            # run all ensembled models on staggered clips, getting frame-level
            # scores averaged across the ensemble
            frame_map = self.predictor.get_overlapping_scores(
                f, initial_start_times=self.ensemble_offsets
            )
            start_times = np.arange(0, frame_map.shape[0]) * self.frame_len
            end_times = start_times + self.frame_len

            # place scores in a dataframe with multi-index of file, start_time, end_time
            df = pd.DataFrame(frame_map, columns=self.predictor.class_names)
            df.index = pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"file": f, "start_time": start_times, "end_time": end_times}
                )
            )
            aggregated_scores.append(df)
        return pd.concat(aggregated_scores)

    def predict_labels(
        self,
        audio_paths,
        min_score=0.8,
        sort_by="time",
    ):
        """generate time-bounded species labels across an audio file

        uses ensembeled models on staggered clips to generate frame-level scores,
        then aggregates contiguous frames above a threshold into 'labels'

        Args:
            audio_paths: str or Path to audio file, or list of str/Path to audio files
            min_score: float, minimum score threshold for a frame to be considered a positive detection
                - scores below the threshold will result in nothing being returned for that species and time window
            sort_by: order of returned elements:
                - 'time': sort by file and start_time (default)
                - 'score': sort by score, descending
                - 'class': sort by class name
                - None: do not sort (faster for large datasets)
        Returns:
            pd.DataFrame with one row per detected species event
        """
        # set detection threshold
        self.predictor.cfg.infer.min_score = min_score

        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]
        aggregated_labels = []

        for f in audio_paths:
            # run all ensembled models on staggered clips, getting frame-level
            # scores averaged across the ensemble
            frame_map = self.predictor.get_overlapping_scores(
                f, initial_start_times=self.ensemble_offsets
            )
            # place scores in a dataframe with multi-index of file, start_time, end_time
            df = self.predictor.get_dataframe(
                score_array=None,
                frame_map=frame_map,
                start_times=None,
                recording_name=f,
            ).rename(columns={"recording": "file", "name": "class"})[
                ["file", "start_time", "end_time", "class", "score"]
            ]
            aggregated_labels.append(df)
        labels = pd.concat(aggregated_labels)
        if sort_by is not None:
            if sort_by == "time":
                labels = labels.sort_values(["file", "start_time"])
            elif sort_by == "score":
                labels = labels.sort_values("score", ascending=False)
            elif sort_by == "class":
                labels = labels.sort_values("class")
        return labels
