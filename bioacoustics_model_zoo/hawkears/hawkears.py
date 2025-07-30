import warnings
import pandas as pd
import cv2
import numpy as np

from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape.preprocess.actions import Action, BaseAction
import opensoundscape
from opensoundscape import Audio, CNN

from bioacoustics_model_zoo.hawkears import hawkears_base_config
from bioacoustics_model_zoo.hawkears.architecture_constructors import get_hgnet
from bioacoustics_model_zoo.utils import download_cached_file, download_file
import torchaudio

import torch

from opensoundscape.ml.cnn import register_model_cls
from opensoundscape.preprocess.actions import register_action_cls

from bioacoustics_model_zoo.hawkears.list_checkpoints import list_hawkears_checkpoints
from bioacoustics_model_zoo.hawkears.architecture_constructors import create_model


@register_action_cls
class HawkEarsSpec(BaseAction):
    """hawkears preprocessing of audio signal to normalized spectrogram

    uses settings from config file's BaseConfig class

    Args:
        cfg: if None, loads BaseConfig from hawkears_base_config module
            - can be a config object from hawkears repo
        low_band: if True, creates low frequency spectrograms for a specialized model
            otherwise creates typical spectrograms for bird classification
        device: torch device (or string name) to use for spectrogram creation
            - eg, 'mps', 'cuda:0', 'cpu'
            - as of April 2024, torchaudio supports cuda but not mps for making spectrograms
            - default 'cpu' is safest best but slowest

    based on:
    https://github.com/jhuus/HawkEars/blob/24bc5a3e031866bc3ff81343bffff83429ee7897/core/audio.py
    """

    def __init__(self, cfg=None, low_band=False, device="cpu"):
        super(HawkEarsSpec, self).__init__()
        self.low_band = low_band

        # use custom config if provided, otherwise default
        if cfg is None:
            cfg = hawkears_base_config.BaseConfig()
        self.cfg = cfg

        # set device (mps/cuda/cpu) to use for spectrogram creation
        self.device = torch.device(device)

        fft_hop = int(
            cfg.audio.segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width
        )

        self.linear_transform = torchaudio.transforms.Spectrogram(
            n_fft=2 * self.cfg.audio.win_length,
            win_length=self.cfg.audio.win_length,
            hop_length=fft_hop,
            power=1,
        )
        self.linear_transform.to(self.device)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.audio.sampling_rate,
            n_fft=2 * self.cfg.audio.win_length,
            win_length=self.cfg.audio.win_length,
            hop_length=fft_hop,
            f_min=self.cfg.audio.min_audio_freq,
            f_max=self.cfg.audio.max_audio_freq,
            n_mels=self.cfg.audio.spec_height,
            power=self.cfg.audio.power,
        )
        self.mel_transform.to(self.device)

    def _normalize(self, spec):
        """normalize values to have max=1"""
        max = spec.max()
        if max > 0:
            spec = spec / max
        return spec.clip(0, 1)

    def _get_raw_spectrogram(self, signal, low_band=False):
        """use config settings to create linear or mel spectrogram"""
        if low_band:  # special settings if we want to detect Ruffed Grouse
            min_audio_freq = self.cfg.audio.low_band_min_audio_freq
            max_audio_freq = self.cfg.audio.low_band_max_audio_freq
            spec_height = self.cfg.audio.low_band_spec_height
            mel_scale = self.cfg.audio.low_band_mel_scale
        else:
            min_audio_freq = self.cfg.audio.min_audio_freq
            max_audio_freq = self.cfg.audio.max_audio_freq
            spec_height = self.cfg.audio.spec_height
            mel_scale = self.cfg.audio.mel_scale

        signal = signal.reshape((1, signal.shape[0]))
        tensor = torch.from_numpy(signal).to(self.device)
        if mel_scale:
            spec = self.mel_transform(tensor).cpu().numpy()[0]
        else:
            spec = self.linear_transform(tensor).cpu().numpy()[0]

        if not mel_scale:
            # clip frequencies above max_audio_freq and below min_audio_freq
            high_clip_idx = int(
                2 * spec.shape[0] * max_audio_freq / self.cfg.audio.sampling_rate
            )
            low_clip_idx = int(
                2 * spec.shape[0] * min_audio_freq / self.cfg.audio.sampling_rate
            )
            spec = spec[:high_clip_idx, low_clip_idx:]
            spec = cv2.resize(
                spec, dsize=(spec.shape[1], spec_height), interpolation=cv2.INTER_AREA
            )

        return spec

    def __call__(self, sample):
        """creates spectrogram, normalizes, casts to torch.tensor"""
        # sample.data will be Audio object. Replace sample.data with torch.tensor of spectrogram.
        spec = self._get_raw_spectrogram(sample.data.samples, low_band=self.low_band)

        # normalize
        spec = self._normalize(spec)

        # reshape if needed (https://github.com/jhuus/HawkEars/blob/f924114ebe6e6f220df74f9fb136f6194f7ac0e8/core/audio.py#L150C17-L153C17)
        spec = spec[:self.cfg.audio.spec_height, :self.cfg.audio.spec_width]
        if spec.shape[1] < self.cfg.audio.spec_width:
            spec = np.pad(spec, ((0, 0), (0, self.cfg.audio.spec_width - spec.shape[1])), 'constant', constant_values=0)
        

        # update the AudioSample's .data in-place
        sample.data = torch.tensor(spec).unsqueeze(0)


import torch
from bioacoustics_model_zoo.utils import download_file, register_bmz_model
from pathlib import Path


class Ensemble(torch.nn.Module):
    """Ensemble of multiple models for classification

    Args:
        models: list of models to use in the ensemble
    """

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        for i, m in enumerate(models):
            self.add_module(f"model_{i}", m)

    def forward(self, x):
        """forward pass through the ensemble, average outputs from across models"""
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


@register_bmz_model
@register_model_cls
class HawkEars(CNN):
    """HawkEars Canadian bird classification CNN v0.1.0

    Hawkears[1] was developed by Jan Huus and is actively maintained on the
    [GitHub repository](https://github.com/jhuus/HawkEars)

    [1] Huus, Jan, et al. 2024 "Hawkears: A Regional, High-Performance Avian
    Acoustic Classifier." SSRN.

    Note that the HawkEars github repo implements various heuristics and filters
    when predicting class presence, while this implementation simply returns the
    ensembled model outputs.

    Note that embed() currently uses embeddings from self.network.model_4, the
    largest of the ensembled inference models. By contrast, in the HawkEars
    GitHub repo, embedding currently uses instead an entirely different model
    with an efficientnet architecture, so results will differ from here.

    Args:
        config: use None for default, or pass a valid object created with the
            HawkEars repo
        force_reload: (bool) default False skips checkpoint downloads if local path already
            exists; True downloads and over-writes existing checkpoint files if ckpt_stem is URL
        load_lowband_model: (bool, optional) default False. if True loads a specialized model for
            detecting low frequency signals such as Ruffed Grouse drumming _instead_ of the
            standard model
        load_embedding_model: bool, optional default False. if True, loads a different model
            architecture that is preferred for embedding and similarity search tasks
        version: default '1.0.8', supports versions 1.0.0-1.0.8 as of July 2025
            - corresponds to release tags in the HawkEars github repo
            - newer versions than v1.0.8 may or may not work without issues
        cache_dir: optional, directory to cache downloaded files
            - typically leave as None to use the default model caching directory
        verbose: optional, bool default False, if True prints information about downloading and
            loading model checkpoints
    Returns:
        model object with methods for generating predictions and embeddings, and trainig


    Example:
    ``` import bioacoustics_model_zoo as bmz
    m=bmz.HawkEars()
    m.predict(['test.wav'],batch_size=64) # returns dataframe of per-class scores
    m.embed(['test.wav']) # returns dataframe of embeddings
    ```
    """

    def __init__(
        self,
        cfg=None,
        force_reload=False,
        version="1.0.8",
        load_lowband_model=False,
        load_embedding_model=False,
        cache_dir=None,
        verbose=False,
    ):
        try:
            import timm
        except Exception as e:
            raise Exception("install the timm package to use HawkEars") from e

        self.is_lowband_model = load_lowband_model
        self.is_embedding_model = load_embedding_model
        if self.is_lowband_model:
            assert (
                not load_embedding_model
            ), "load_lowband_model and load_embedding_model cannot both be True"
            self.name = "hawkears_low_band"
        elif self.is_embedding_model:
            self.name = "hawkears_embedding"
        else:
            self.name = "hawkears"

        self.version = version

        # use custom config if provided, otherwise default
        if cfg is None:
            cfg = hawkears_base_config.BaseConfig()
        self.cfg = cfg

        # load appropriate checkpoints based on whether we want to
        # creat the lowband model, embedding model, or typical bird species ensemble
        ensemble_checkpoints = list_hawkears_checkpoints(
            self.version,
            lowband=self.is_lowband_model,
            embedding=self.is_embedding_model,
        )
        all_models = []
        class_codes = None
        classes = None

        for ckpt_path in ensemble_checkpoints:
            # download model if URL, otherwise find it at local path:
            if str(ckpt_path).startswith("http"):
                # will skip download if file exists; to force re-download, delete existing checkpoints
                if verbose:
                    print("Downloading model from URL...")
                filename = Path(ckpt_path).name
                model_path = download_cached_file(
                    ckpt_path,
                    filename=filename,
                    model_name=self.name,
                    model_version=self.version,
                    cache_dir=cache_dir,
                    redownload_existing=force_reload,
                )
            else:
                assert Path(ckpt_path).exists(), f"Checkpoint not found at {ckpt_path}"
                model_path = ckpt_path
            model_path = str(Path(model_path).resolve())  # get absolute path as string
            assert Path(model_path).exists(), f"Model path {model_path} does not exist"
            if verbose:
                print(f"Loading model from local checkpoint {model_path}...")
            mdict = torch.load(model_path, map_location=torch.device("cpu"))

            # create architecture and load saved weights
            model_name = mdict["hyper_parameters"]["model_name"]
            m_classes = mdict["hyper_parameters"]["train_class_names"]
            m_class_codes = mdict["hyper_parameters"]["train_class_codes"]

            if class_codes is None:
                class_codes = m_class_codes
            if classes is None:
                classes = m_classes

            model = create_model(model_name, num_classes=len(classes))

            # remove 'base_model' prefix from state dict keys to match our architecture
            state_dict = {
                k.replace("base_model.", ""): v for k, v in mdict["state_dict"].items()
            }
            # load checkpoint weights into torch module
            model.load_state_dict(state_dict)

            # add this model to the list of models to use as an ensemble
            all_models.append(model)

        # initialize a module that runs the ensemble and averages outputs
        # if there is more than one model in the model list
        # otherwise use the model directly as the .network property
        if len(all_models) > 1:
            arch = Ensemble(all_models)
        else:
            arch = all_models[0]

        # select default layers for embedding/cam/classifier replacement
        # TODO: likely to break when ensemble changes
        if self.is_embedding_model:
            arch.cam_layer = "bn2"
            arch.embedding_layer = "global_pool"
            arch.classifier_layer = "classifier"
        elif self.is_lowband_model:
            arch.cam_layer = "global_pool"
            # dla doesn't flatten dimensions until classification!
            arch.embedding_layer = None
            arch.classifier_layer = "flatten"
        else:  # typical ensemble
            arch.cam_layer = "model_0.final_conv"  # default layer for activation maps
            # default layer for embeddings
            arch.embedding_layer = "model_0.head.global_pool"
            # layer accessed w/ `self.classifier`
            arch.classifier_layer = "model_0.head.fc"

        # initialize the CNN object with this architecture and class list
        # use 3s duration and expected sample shape for HawkEars
        super(HawkEars, self).__init__(
            arch,
            classes=classes,
            sample_duration=cfg.audio.segment_len,
            sample_shape=[cfg.audio.spec_height, cfg.audio.spec_width, 1],
        )
        self.class_codes = class_codes
        """4-letter alpha codes (or similar for non-birds) corresponding to self.classes"""

        # compose the preprocessing pipeline:
        # load audio with 3s framing; extend to 3s if needed
        # create spectrogram based on config values, using same functions as HawkEars to ensure same results
        # normalize spectrogram to max=1
        pre = AudioAugmentationPreprocessor(
            sample_duration=cfg.audio.segment_len, sample_rate=cfg.audio.sampling_rate
        )
        pre.insert_action(
            action_index="extend",
            action=Action(
                Audio.extend_to, is_augmentation=False, duration=cfg.audio.segment_len
            ),
        )

        # note: can specify pre.pipeline.to_spec.device = 'cuda' to use cuda for spectrogram creation
        # but mps does not support spectrogram creation as of April 2024
        pre.insert_action(
            action_index="to_spec",
            action=HawkEarsSpec(cfg=cfg, low_band=load_lowband_model),
        )
        self.preprocessor = pre

        # set the classifier learning rate to be higher than the feature extractor
        # Note: I have not experimented to find optimal learning rates or relative learning rates
        self.optimizer_params["classifier_lr"] = 0.01
        self.optimizer_params["kwargs"]["lr"] = 0.001

    def freeze_feature_extractor(self):
        # the hgmodels all have .fc as the final classification layer
        # so we can freeze all except each model's .fc
        if self.is_lowband_model:
            return self.freeze_layers_except([self.network.fc])
        elif self.is_embedding_model:
            return self.freeze_layers_except(self.network.classifier)
        else:  # ensemble of models with .head.fc layers
            return self.freeze_layers_except([m.head.fc for m in self.network.models])

    @classmethod
    def load(cls, path, ckpt_stem=None):
        """reload object after saving to file with .save()

        Args:
            path: path to file saved using .save()
            ckpt_stem: path or URL containing checkpoint files for all ensembled models
                (pass a local path if you have already downloaded the checkpoints; if None,
                will check if present in current directory otherwise download from GitHub)

        Returns:
            new HawkEars instance

        Note: Note that if you used pickle=True when saving, the model object
        might not load properly across different versions of OpenSoundscape.

        Note: modifications to preprocessor are not retained when
        saving/loading, unless .pickle format is used
        """
        loaded_content = torch.load(path)

        opso_version = (
            loaded_content.pop("opensoundscape_version")
            if isinstance(loaded_content, dict)
            else loaded_content.opensoundscape_version
        )
        if opso_version != opensoundscape.__version__:
            warnings.warn(
                f"Model was saved with OpenSoundscape version {opso_version}, "
                f"but you are currently using version {opensoundscape.__version__}. "
                "This might not be an issue but you should confirm that the model behaves as expected."
            )

        if isinstance(loaded_content, dict):
            # initialize with random weights is not currently supported, so init as normal
            # from checkpoints (this provides the correct architectures)
            model = cls(ckpt_stem=ckpt_stem)
            # update all classifier heads based on the class list
            model.change_classes(loaded_content["classes"])
            # load up the weights and instantiate from dictionary keys
            # including the weights of custom classifier heads
            state_dict = loaded_content.pop("weights")
            model.network.load_state_dict(state_dict)
        else:
            model = loaded_content  # entire pickled object, not dictionary

        return model

    def recreate_clf(self):
        # use appropriate output dimensions (`len(self.classes)`) for each sub-model's fc layer
        if self.is_lowband_model:
            raise NotImplementedError(
                """try manually adapting the classifier of the lowband DLA model, it is a conv2d 
                (see https://github.com/jhuus/HawkEars/blob/f924114ebe6e6f220df74f9fb136f6194f7ac0e8/model/main_model.py#L159)"""
            )
        elif self.is_embedding_model:
            self.network.classifier = torch.nn.Linear(
                self.network.classifier.in_features, len(self.classes)
            )
        else:  # ensembole of models with .head.fc layers
            for submodel in self.network.models:
                # initializes new FC layer with random weights
                submodel.head.fc = torch.nn.Linear(
                    submodel.head.fc.in_features, len(self.classes)
                )
        self.network.to(self.device)

    @property
    def classifier_params(self):
        """return list of parameters for the classification layers from all ensembled models"""

        if self.is_lowband_model:
            raise NotImplementedError(
                """Note that this model has an unusual classifier head: conv2D then pooling"""
            )
        elif self.is_embedding_model:
            return self.network.classifier.parameters()
        else:  # ensembole of models with .head.fc layers
            clf_params = []
            for submodel in self.network.models:
                clf_params.extend(submodel.head.fc.parameters())
            return clf_params

    def change_classes(self, new_classes):
        """modify the classification heads of all ensembled models to match new class list

        Args:
            new_classes: list of class names to use for classification
        Effects:
            - updates self.classes and torch metrics to match new classes
            - changes output layer sizes to match len(classes)
            - initializes fc layers of each ensembled sub-model with random weights
            - self.class_codes is set to None, as original HawkEars alpha codes no
                longer match self.classes
        """
        self.classes = new_classes
        self.class_codes = None  # alpha codes no longer known
        self._init_torch_metrics()
        self.recreate_clf()


@register_bmz_model
@register_model_cls
class HawkEars_Low_Band(HawkEars):
    """version of HawkEars model specifically designed for Ruffed Grouse and Spruce Grouse"""

    def __init__(
        self,
        cfg=None,
        force_reload=False,
        version="1.0.8",
        cache_dir=None,
        verbose=False,
    ):
        super().__init__(
            cfg=cfg,
            force_reload=force_reload,
            load_lowband_model=True,
            version=version,
            cache_dir=cache_dir,
            verbose=verbose,
        )


@register_bmz_model
@register_model_cls
class HawkEars_Embedding(HawkEars):
    """version of HawkEars model specifically designed for embedding tasks"""

    def __init__(
        self,
        cfg=None,
        force_reload=False,
        version="1.0.8",
        cache_dir=None,
        verbose=False,
    ):
        super().__init__(
            cfg=cfg,
            force_reload=force_reload,
            load_embedding_model=True,
            version=version,
            cache_dir=cache_dir,
            verbose=verbose,
        )


@register_bmz_model
@register_model_cls
class HawkEars_v010(HawkEars):
    """HawkEars Canadian bird classification CNN v0.1.0

    Hawkears[1] was developed by Jan Huus and is actively maintained on the
    [GitHub repository](https://github.com/jhuus/HawkEars)

    [1] Huus, Jan, et al. 2024 "Hawkears: A Regional, High-Performance Avian
    Acoustic Classifier." SSRN.

    Note that the HawkEars github repo implements various heuristics and filters
    when predicting class presence, while this implementation simply returns the
    ensembled model outputs.

    Note that embed() currently uses embeddings from self.network.model_4, the
    largest of the ensembled inference models. By contrast, in the HawkEars
    GitHub repo, embedding currently uses instead an entirely different model
    with an efficientnet architecture, so results will differ from here.

    Args:
        config: use None for default, or pass a valid object created with the
            HawkEars repo
        ckpt_stem: path or URL containing checkpoint files for all ensembled models
            (pass a local path if you have already downloaded the checkpoints; use the
            default to download v0.1.0 checkpoints from GitHub)
        force_reload: (bool) default False skips checkpoint downloads if local path already
            exists; True downloads and over-writes existing checkpoint files if ckpt_stem is URL
        version: only v0.1.0 is currently supported - updated versions coming soon
        cache_dir: optional, directory to cache downloaded files
            - typically leave as None to use the default model caching directory

    Returns:
        model object with methods for generating predictions and embeddings, and trainig


    Example:
    ``` import bioacoustics_model_zoo as bmz
    m=bmz.HawkEars()
    m.predict(['test.wav'],batch_size=64) # returns dataframe of per-class scores
    m.embed(['test.wav']) # returns dataframe of embeddings
    ```
    """

    def __init__(
        self,
        cfg=None,
        ckpt_stem=None,
        force_reload=False,
        cache_dir=None,
    ):
        self.version = "0.1.0"
        self.name = "hawkears_v010"
        # use custom config if provided, otherwise default
        if cfg is None:
            cfg = hawkears_base_config.BaseConfig()
        self.cfg = cfg

        if ckpt_stem is None:
            ckpt_stem = (
                "https://github.com/jhuus/HawkEars/raw/refs/tags/0.1.0/data/ckpt"
            )
        elif not str(ckpt_stem).startswith("http"):
            ckpt_stem = Path(ckpt_stem).resolve()  # local path: convert to full path

        # HawkEars v0.1.0 uses 5 HGNETs in an ensemble
        all_checkpoints = [f"{ckpt_stem}/hgnet{i}.ckpt" for i in range(1, 6)]
        all_models = []
        class_codes = None
        classes = None

        for ckpt_path in all_checkpoints:
            # download model if URL, otherwise find it at local path:
            if str(ckpt_path).startswith("http"):
                # will skip download if file exists; to force re-download, delete existing checkpoints
                print("Downloading model from URL...")
                filename = Path(ckpt_path).name
                model_path = download_cached_file(
                    ckpt_path,
                    filename=filename,
                    model_name=self.name,
                    model_version=self.version,
                    cache_dir=cache_dir,
                    redownload_existing=force_reload,
                )
            else:
                assert Path(ckpt_path).exists(), f"Checkpoint not found at {ckpt_path}"
                model_path = ckpt_path
            model_path = str(Path(model_path).resolve())  # get absolute path as string
            assert Path(model_path).exists(), f"Model path {model_path} does not exist"
            print(f"Loading model from local checkpoint {model_path}...")
            mdict = torch.load(model_path, map_location=torch.device("cpu"))

            model_name = mdict["hyper_parameters"]["model_name"]

            m_classes = mdict["hyper_parameters"]["train_class_names"]
            m_class_codes = mdict["hyper_parameters"]["train_class_codes"]

            if class_codes is None:
                class_codes = m_class_codes
            if classes is None:
                classes = m_classes

            # Note: if a new tag has an ensemble of models uses a different
            # architecture, we will need to change the model constructor logic
            hgshape = model_name.split("_")[-1]
            model = get_hgnet(hgshape, num_classes=len(classes))
            # remove 'base_model' prefix from state dict keys to match our architecture
            state_dict = {
                k.replace("base_model.", ""): v for k, v in mdict["state_dict"].items()
            }
            # load checkpoint weights into torch module
            model.load_state_dict(state_dict)

            all_models.append(model)

        # initialize a module that runs the ensemble and averages outputs
        arch = Ensemble(all_models)

        # TODO: think about what to set here, just picked the largest model for now
        arch.cam_layer = "model_4.stages.3"  # default layer for activation maps
        arch.embedding_layer = "model_4.head.flatten"  # default layer for embeddings
        arch.classifier_layer = "model_4.head.fc"  # layer accessed w/ `self.classifier`

        # initialize the CNN object with this architecture and class list
        # use 3s duration and expected sample shape for HawkEars
        super(HawkEars, self).__init__(
            arch,
            classes=classes,
            sample_duration=cfg.audio.segment_len,
            sample_shape=[cfg.audio.spec_height, cfg.audio.spec_width, 1],
        )
        self.class_codes = class_codes
        """4-letter alpha codes (or similar for non-birds) corresponding to self.classes"""

        # compose the preprocessing pipeline:
        # load audio with 3s framing; extend to 3s if needed
        # create spectrogram based on config values, using same functions as HawkEars to ensure same results
        # normalize spectrogram to max=1
        pre = AudioAugmentationPreprocessor(
            sample_duration=cfg.audio.segment_len, sample_rate=cfg.audio.sampling_rate
        )
        pre.insert_action(
            action_index="extend",
            action=Action(
                Audio.extend_to, is_augmentation=False, duration=cfg.audio.segment_len
            ),
        )

        # note: can specify pre.pipeline.to_spec.device = 'cuda' to use cuda for spectrogram creation
        # but mps does not support spectrogram creation as of April 2024
        pre.insert_action(action_index="to_spec", action=HawkEarsSpec(cfg=cfg))
        self.preprocessor = pre

        # set the classifier learning rate to be higher than the feature extractor
        # Note: I have not experimented to find optimal learning rates or relative learning rates
        self.optimizer_params["classifier_lr"] = 0.01
        self.optimizer_params["kwargs"]["lr"] = 0.001

    def freeze_feature_extractor(self):
        # the hgmodels all have .fc as the final classification layer
        # so we can freeze all except each model's .fc
        return self.freeze_layers_except([m.head.fc for m in self.network.models])

    @classmethod
    def load(cls, path, ckpt_stem=None):
        """reload object after saving to file with .save()

        Args:
            path: path to file saved using .save()
            ckpt_stem: path or URL containing checkpoint files for all ensembled models
                (pass a local path if you have already downloaded the checkpoints; if None,
                will check if present in current directory otherwise download from GitHub)

        Returns:
            new HawkEars instance

        Note: Note that if you used pickle=True when saving, the model object
        might not load properly across different versions of OpenSoundscape.

        Note: modifications to preprocessor are not retained when
        saving/loading, unless .pickle format is used
        """
        loaded_content = torch.load(path)

        opso_version = (
            loaded_content.pop("opensoundscape_version")
            if isinstance(loaded_content, dict)
            else loaded_content.opensoundscape_version
        )
        if opso_version != opensoundscape.__version__:
            warnings.warn(
                f"Model was saved with OpenSoundscape version {opso_version}, "
                f"but you are currently using version {opensoundscape.__version__}. "
                "This might not be an issue but you should confirm that the model behaves as expected."
            )

        if isinstance(loaded_content, dict):
            # initialize with random weights is not currently supported, so init as normal
            # from checkpoints (this provides the correct architectures)
            model = cls(ckpt_stem=ckpt_stem)
            # update all classifier heads based on the class list
            model.change_classes(loaded_content["classes"])
            # load up the weights and instantiate from dictionary keys
            # including the weights of custom classifier heads
            state_dict = loaded_content.pop("weights")
            model.network.load_state_dict(state_dict)
        else:
            model = loaded_content  # entire pickled object, not dictionary

        return model

    @property
    def ensemble_list(self):  # TODO: could this just be self.network.models instead?
        return [
            self.network.model_0,
            self.network.model_1,
            self.network.model_2,
            self.network.model_3,
            self.network.model_4,
        ]

    def recreate_clf(self):
        # use appropriate output dimensions (`len(self.classes)`) for each sub-model's fc layer
        for submodel in self.ensemble_list:
            # initializes new FC layer with random weights
            submodel.head.fc = torch.nn.Linear(2048, len(self.classes))
        self.network.to(self.device)

    @property
    def classifier_params(self):
        """return list of parameters for the classification layers from all ensembled models"""
        clf_params = []
        for submodel in self.ensemble_list:
            clf_params.extend(submodel.head.fc.parameters())
        return clf_params

    def change_classes(self, new_classes):
        """modify the classification heads of all ensembled models to match new class list

        Args:
            new_classes: list of class names to use for classification
        Effects:
            - updates self.classes and torch metrics to match new classes
            - changes output layer sizes to match len(classes)
            - initializes fc layers of each ensembled sub-model with random weights
            - self.class_codes is set to None, as original HawkEars alpha codes no
                longer match self.classes
        """
        self.classes = new_classes
        self.class_codes = None  # alpha codes no longer known
        self._init_torch_metrics()
        self.recreate_clf()
