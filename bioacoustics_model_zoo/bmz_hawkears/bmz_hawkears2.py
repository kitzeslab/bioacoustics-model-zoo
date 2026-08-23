"""Run HawkEars v2.x model ensembles in Sound Event Detection mode using BriteKit"""

from typing import Literal
import zipfile
import numpy as np
from copy import deepcopy
import functools

from pathlib import Path
from opensoundscape.ml.cnn import register_model_cls

from bioacoustics_model_zoo.utils import register_bmz_model
from bioacoustics_model_zoo.cache import get_model_cache_dir
import yaml
import pandas as pd

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

    _class_name_map = {
        "common": "names",
        "alpha": "codes",
        "scientific": "alt_names",
        "eBird": "alt_codes",
    }

    def _checkpoints_downloaded(self) -> bool:
        """Check if HawkEars2 model checkpoints are downloaded in the cache directory

        Args:
            cache_dir: Path, path to cache directory. If None, uses default
                cache directory for bioacoustics_model_zoo
        Returns:
            bool, True if checkpoints are downloaded, False otherwise
        """
        # Note: this hard-codes assumptions about the structure and names of ckpt folders.
        # it matches HawkEars v2.2 and 2.3, but I have requested hawkears helper function
        # to avoid this becoming stale in future versions:
        # https://github.com/jhuus/HawkEars/issues/5
        for ckpt_type in "ckpt", "ckpt-low-band":
            ckpt_folder = Path(self.cache_dir) / "data" / ckpt_type
            if not ckpt_folder.exists() or len(list(ckpt_folder.glob("*.ckpt"))) == 0:
                return False
        return True

    @property
    def _main_ckpts_dir(self) -> Path:
        """Get the path to the main HawkEars2 model checkpoints directory

        Returns:
            Path, path to main HawkEars2 model checkpoints directory
        """
        return self.cache_dir / "data" / "ckpt"

    @property
    def _lowband_ckpts_dir(self) -> Path:
        """Get the path to the low-band HawkEars2 model checkpoints directory

        Returns:
            Path, path to low-band HawkEars2 model checkpoints directory
        """
        return self.cache_dir / "data" / "ckpt-low-band"

    def __init__(self, cfg_path=None, force_redownload=False, cache_dir=None):
        """Initialize HawkEars2 with config file

        NOTE: the exact version of HawkEars model (eg., 2.2.0) will match the
        version of the installed hawkears package. Make sure you pip install the
        package version matching the desired model version
        (e.g., `pip install hawkears==2.2.0`).

        NOTE: this implementation does not support training or generation of embeddings. It currently also
        does not support the generation of scores for fixed-length clips, it only generates 'labels'.

        Downloads the HawkEars2 model checkpoints to the bioacoustics_model_zoo cache directory if needed,
        and initializes the Analyzer with the config file. To use a custom config, create a copy of
        the default config file and modify as needed: ``` from
        bioacoustics_model_zoo.hawkears import HawkEars2 he2 =
        HawkEars2(cfg_path="my_config.yaml") he2.save_config("my_config.yaml")
        # save the current config to a file ```

        Args:
            cfg_path: str, path to config yaml file. If None, uses default config
            force_redownload: bool, if True, forces re-download of model checkpoints
            cache_dir: str, path to cache directory. If None, uses default cache
                directory for bioacoustics_model_zoo. Specifying a custom cache dir is
                not recommended unless you need advanced control (e.g., multi-user server).

        Methods:

        """
        try:
            import hawkears
        except ImportError:
            raise ImportError(
                "HawkEars2 requires the hawkears package to be installed (`pip install hawkears`)."
            )

        # currently, HawkEars model version is always the same as the installed hawkears package version
        self.version = hawkears.__version__
        print(
            f"HawkEars version: {self.version} (model version mirrors the installed hawkears package version)"
        )
        if cache_dir is None:
            cache_dir = get_model_cache_dir("hawkears2", self.version)
        self.cache_dir = Path(cache_dir)

        # download checkpoints if needed or if force_redownload is True
        # Note: just checks for >=1 ckpt of each type, may not be all available ckpts
        # but since it downloada a zip then unpacks, this is decently robust
        if force_redownload or not self._checkpoints_downloaded():
            hawkears.core.initializer.initialize(cache_dir)

        # initialize the Analyzer (ml model runner) with config file
        # if cfg_path=None, uses default config file in package
        self.cfg = hawkears.core.config_loader.get_config(
            cfg_path=cfg_path, data_root=Path(cache_dir)
        )
        self.cfg.infer.label_field = "names"  # default to common names

    def save_config(self, path):
        """Save the current config to a YAML file"""
        from omegaconf import OmegaConf

        with open(path, "w") as f:
            OmegaConf.save(self.cfg, f)

    @property
    def classes(self):
        """return list of class names, matching the current label_field setting (common/alpha/scientific/eBird)"""
        import hawkears

        analyzer = hawkears.core.analyzer.Analyzer(self.cfg)
        name_type = self.cfg.infer.label_field[:-1]  # remove trailing 's'
        return [getattr(c, name_type) for c in analyzer.class_mgr.all_classes()]

    def set_class_name_type(
        self, name_type: Literal["common", "alpha", "scientific", "eBird"]
    ):
        """Select common/scientific/alpha/eBird as output class names

        Outputs such as clip predictions and species detection labels will use the selected
        naming convention for species names. The default is "common" (common names).

        Args:
            name_type: str, one of "common", "alpha", "scientific", "eBird"
        """
        self.cfg.infer.label_field = self._class_name_map[name_type]

    def label(
        self,
        files,
        threshold=None,
        output_path=None,
        num_threads=None,
        max_models=None,
        class_names=None,
        date=None,
        region=None,
        lat=None,
        lon=None,
        recurse=True,
        include_lowband_classifier=True,
        quiet=False,
    ):
        """Generate time-bounded species labels across an audio file or folder of files

        Note: this function is a wrapper around HawkEars Analyzer.run() and includes all post-processing,
        such as region and date based filtering and score hueristics.
        By contrast, .predict() and .predict_frames() simply return the averaged outputs of ensembled models.

        Uses ensembled models on staggered clips to generate frame-level scores,
        which are then aggregated to produce time-bounded species labels.

        All settings from self.cfg unless over-ridden by arguments

        Args:
            file_or_folder: str or Path to audio file, or list of str/Path to audio files
            threshold: float, minimum score threshold for a frame to be considered a positive detection
                - scores below the threshold will result in nothing being returned for that species and time window
            output_path: str or Path, path to folder where output files will be saved. If None, uses a temp folder
            num_threads: int, number of threads to use for inference. If None, uses config
            max_models: int, maximum number of models to use for inference. If None, uses config
            class_names: str, one of "common", "alpha", "scientific", "eBird" to select which class names to use in output
            date (str, optional): Date as YYYYMMDD, MMDD, or 'file'. Specifying 'file' extracts the date from the file name.
            region (str, optional): eBird region code, e.g. 'CA-AB' for Alberta. Use as an alternative to latitude/longitude.
                - Note that only a subset of regions are supported by HawkEars occurrence db
            lat (float, optional): Latitude.
            lon (float, optional): Longitude.
            recurse: bool, whether to recursively search subfolders for audio files
            include_lowband_classifier: bool, whether to also use the low-band model for Ruffed Grouse and Spruce Grouse detection
            quiet: bool, whether to suppress progress output
        """
        import hawkears

        cfg = deepcopy(self.cfg)

        # update analyzer config with user-specified settings:
        cfg.hawkears.low_band_classifier = include_lowband_classifier
        if threshold is not None:
            cfg.infer.min_score = threshold
        if num_threads is not None:
            cfg.infer.num_threads = num_threads
        if max_models is not None:
            cfg.infer.max_models = max_models
        if class_names is not None:
            cfg.infer.label_field = self._class_name_map[class_names]
        if region is not None:
            cfg.hawkears.region = region
            # TODO: region is not filtering outputs
        if lat is not None:
            cfg.hawkears.latitude = lat
        if lon is not None:
            cfg.hawkears.longitude = lon

        # use temp file if output_path is None
        if output_path is None:
            import tempfile

            output_path = tempfile.mkdtemp()

        # note: simply modifying existing analyzer's cfg doesn't work well (eg changing .cfg.infer.label_field results in no outputs)
        # so we create a fresh Analyzer at runtime with the desired cfg.
        # todo: can pass class_list but only accepts common names. Could convert here.
        analyzer = hawkears.core.analyzer.Analyzer(cfg)

        if region is not None:
            # check if this region is included in the occurrence db, if not raise error
            from hawkears.core.occurrence_manager import OccurrenceManager

            om = OccurrenceManager(analyzer.cfg, analyzer.class_mgr, [])
            if not om._region_has_data(region):
                raise ValueError(
                    f"Region {region} is not included in HawkEars occurrence database and cannot be used to filter detections."
                )
        # analyzer saves outputs in self.result_dataframes
        analyzer.run(
            files,
            output_path,
            rtypes=[],
            date=date,
            start_seconds=0,
            recurse=recurse,
            top=False,
            quiet=quiet,
            return_results=True,
            progress_callback=None,
            cancellation_callback=None,
        )

        # aggregate results from analyzer into output dataframe
        all_results = []
        for f, df in analyzer.result_dataframes:
            df["file"] = f  # copy full or relative path into df
            df.rename(columns={"name": "class"}, inplace=True)
            df = df[["file", "start_time", "end_time", "class", "score"]]
            all_results.append(df)

        return pd.concat(all_results, ignore_index=True)

    def _get_predictors(self, device=None, include_lowband_classifier=True):
        """Initialize and return the main and low-band britekit predictor objects

        Args:
            device: str, device to use for inference (e.g., "cpu", "cuda:0", "mps").
                - if None, auto-selects GPU when available
            include_lowband_classifier: bool, whether to also initialize the low-band predictor for Ruffed Grouse and Spruce Grouse detection

        Returns:
            predictor: britekit.Predictor, main HawkEars2 predictor
            lowband_predictor: britekit.Predictor, low-band predictor for Ruffed Grouse and Spruce Grouse detection
                (None if include_lowband_classifier is False)
            class_replacement_map: dict, mapping from low-band model class indices to main model class indices for Ruffed Grouse and Spruce Grouse
                (None if include_lowband_classifier is False)
        """
        import britekit

        predictor = britekit.Predictor(
            self._main_ckpts_dir, device=device, cfg=self.cfg
        )
        predictor.cfg = deepcopy(self.cfg)

        if include_lowband_classifier:
            lowband_predictor = britekit.Predictor(
                self._lowband_ckpts_dir, device=device, cfg=self.cfg
            )
            lowband_predictor.cfg = deepcopy(self.cfg)
            lowband_predictor.cfg.infer.label_field = "names"
            # specify class indices to replace with lowband scores
            # maps lowband model's class idx to main models (self.classes) idx
            class_replacement_map = {
                lowband_idx: self.classes.index(c)
                for lowband_idx, c in enumerate(lowband_predictor.class_names)
                if not c in ["Other", "Speech"]
            }
        else:
            lowband_predictor = None
            class_replacement_map = None

        return predictor, lowband_predictor, class_replacement_map

    def predict(self, files, include_lowband_classifier=True, device=None):
        """Generate scores for each class on fixed-length clips across all audio

        Does not apply date/location filtering or heuristics (use label() for full HawkEars post-processing)

        Args:
            audio_paths: str or Path to audio file, or list of str/Path to audio
            include_lowband_classifier: bool, whether to also use the low-band model for Ruffed Grouse and Spruce Grouse detection
            device: str, device to use for inference (e.g., "cpu", "cuda:0", "mps"). If None, auto-selects GPU when available

        Returns:
            pd.DataFrame with one row per clip and a multi-index of file, start_time, end_time and columns for each class with prediction scores
        """
        if isinstance(files, (str, Path)):
            files = [files]

        # initialize predictor and low_band predictor
        predictor, lowband_predictor, class_replacement_map = self._get_predictors(
            device=device, include_lowband_classifier=include_lowband_classifier
        )

        aggregated_scores = []
        for f in files:
            # run all ensembled models on the same clips, getting clip-level
            # scores averaged across the ensemble
            clip_scores, _, starts = predictor.get_recording_scores(f)
            start_times = np.array(starts)
            end_times = start_times + predictor.cfg.audio.spec_duration

            # if lowband model is enabled, update the clip scores for Ruffed Grouse and Spruce Grouse
            # (specifically, all classes except "Other" and "Speech")
            if include_lowband_classifier:
                # this method performs traditional segment-level (3s clip) inference
                low_band_scores, _, _ = lowband_predictor.get_recording_scores(f)
                # update main scores: use the max clip-wise max score for classes in low-band model
                num_clips = min(clip_scores.shape[0], low_band_scores.shape[0])
                for lowband_cls_idx, main_cls_idx in class_replacement_map.items():
                    low_band_cls_scores = low_band_scores[:num_clips, lowband_cls_idx]
                    clip_scores[:num_clips, main_cls_idx] = np.maximum(
                        clip_scores[:num_clips, main_cls_idx], low_band_cls_scores
                    )

            # place scores in a dataframe with multi-index of file, start_time, end_time
            df = pd.DataFrame(clip_scores, columns=predictor.class_names)
            df.index = pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"file": f, "start_time": start_times, "end_time": end_times}
                )
            )
            aggregated_scores.append(df)
        return pd.concat(aggregated_scores)

    @property
    def frame_len(self):
        return 1 / self.cfg.train.sed_fps

    def predict_frames(
        self,
        files,
        include_lowband_classifier=True,
        device=None,
    ):
        """Generate frame-level species prediction scores

        Does not apply date/location filtering or heuristics (use label() for full HawkEars post-processing)

        Args:
            audio_paths: str or Path to audio file, or list of str/Path to audio files

        Returns:
            pd.DataFrame with one row per (0.25-second) frame and a multi-index of
            file, start_time, end_time and columns for each class with
            prediction scores
        """
        if isinstance(files, (str, Path)):
            files = [files]

        # initialize predictor and low_band predictor
        predictor, lowband_predictor, class_replacement_map = self._get_predictors(
            device=device, include_lowband_classifier=include_lowband_classifier
        )

        aggregated_scores = []
        ensemble_offsets = get_start_time_offsets(
            len(predictor.models), predictor.cfg.audio.spec_duration
        )

        for f in files:
            # run all ensembled models on staggered clips, getting frame-level
            # scores averaged across the ensemble
            frame_map = predictor.get_overlapping_scores(
                f, initial_start_times=ensemble_offsets
            )
            start_times = np.arange(0, frame_map.shape[0]) * self.frame_len
            end_times = start_times + self.frame_len

            # if lowband model is enabled, update the clip scores for Ruffed Grouse and Spruce Grouse
            if include_lowband_classifier:
                _, low_band_frames, _ = lowband_predictor.get_recording_scores(f)
                # update main scores: use the max clip-wise max score for low-band model classes
                num_frames = min(frame_map.shape[0], low_band_frames.shape[0])
                for lowband_cls_idx, main_cls_idx in class_replacement_map.items():
                    low_band_cls_frames = low_band_frames[:num_frames, lowband_cls_idx]
                    frame_map[:num_frames, main_cls_idx] = np.maximum(
                        frame_map[:num_frames, main_cls_idx], low_band_cls_frames
                    )

            # place scores in a dataframe with multi-index of file, start_time, end_time
            df = pd.DataFrame(frame_map, columns=predictor.class_names)
            df.index = pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"file": f, "start_time": start_times, "end_time": end_times}
                )
            )
            aggregated_scores.append(df)
        return pd.concat(aggregated_scores)
