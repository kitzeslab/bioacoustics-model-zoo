import warnings
import pandas as pd
import cv2

from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape.preprocess.actions import Action, BaseAction
import opensoundscape
from opensoundscape import Audio, CNN

from . import hawkears_base_config
from .architecture_constructors import get_hgnet
import torchaudio


class HawkEarsSpec(BaseAction):
    """hawkears preprocessing of audio signal to normalized spectrogram

    uses settings from config file's BaseConfig class

    Args:
        cfg: if None, loads BaseConfig from hawkears_base_config module
            - can be a config object from hawkears repo
        device: torch device (or string name) to use for spectrogram creation
            - eg, 'mps', 'cuda:0', 'cpu'
            - as of April 2024, torchaudio supports cuda but not mps for making spectrograms
            - default 'cpu' is safest best but slowest

    based on:
    https://github.com/jhuus/HawkEars/blob/24bc5a3e031866bc3ff81343bffff83429ee7897/core/audio.py
    """

    def __init__(self, cfg=None, device="cpu"):
        super(HawkEarsSpec, self).__init__()

        # use custom config if provided, otherwise default
        if cfg is None:
            cfg = hawkears_base_config.BaseConfig()
        self.cfg = cfg

        # set device (mps/cuda/cpu) to use for spectrogram creation
        self.device = device

        self.linear_transform = torchaudio.transforms.Spectrogram(
            n_fft=2 * self.cfg.audio.win_length,
            win_length=self.cfg.audio.win_length,
            hop_length=self.cfg.audio.hop_length,
            power=1,
        ).to(self.device)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg.audio.sampling_rate,
            n_fft=2 * self.cfg.audio.win_length,
            win_length=self.cfg.audio.win_length,
            hop_length=self.cfg.audio.hop_length,
            f_min=self.cfg.audio.min_audio_freq,
            f_max=self.cfg.audio.max_audio_freq,
            n_mels=self.cfg.audio.spec_height,
            power=self.cfg.audio.power,
        ).to(self.device)

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
        spec = self._get_raw_spectrogram(sample.data.samples)

        # normalize
        spec = self._normalize(spec)

        # update the AudioSample's .data in-place
        sample.data = torch.tensor(spec).unsqueeze(0)


import torch
from ..utils import download_github_file
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
        """forward pass through the ensemble"""
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


class HawkEars(CNN):
    """HawkEars bird classification CNN v0.1.0

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
            default URL to download checkpoints from GitHub)
        force_reload: (bool) default False skips checkpoint downloads if local path already
            exists; True downloads and over-writes existing checkpoint files if ckpt_stem is URL

    Example:
    ``` import torch
    m=torch.hub.load('kitzeslab/bioacoustics-model-zoo',
    'HawkEars',trust_repo=True) m.predict(['test.wav'],batch_size=64) # returns
    dataframe of per-class scores m.embed(['test.wav']) # returns dataframe of
    embeddings
    ```
    """

    def __init__(
        self,
        cfg=None,
        ckpt_stem="https://github.com/jhuus/HawkEars/raw/refs/tags/0.1.0/data/ckpt/",
        force_reload=False,
    ):
        # use custom config if provided, otherwise default
        if cfg is None:
            cfg = hawkears_base_config.BaseConfig()
        self.cfg = cfg

        all_checkpoints = [f"{ckpt_stem}hgnet{i}.ckpt" for i in range(1, 6)]
        all_models = []
        classes = None
        class_codes = None

        for ckpt_path in all_checkpoints:
            # download model if URL, otherwise find it at local path:
            if ckpt_path.startswith("http"):
                # will skip download if file exists; to force re-download, delete existing checkpoints
                print("Downloading model from URL...")
                model_path = download_github_file(
                    ckpt_path, redownload_existing=force_reload
                )
            else:
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
            else:
                assert (
                    class_codes == m_class_codes
                ), "Class codes do not match across models"

            if classes is None:
                classes = m_classes
            else:
                assert classes == m_classes, "Class names do not match across models"

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
        arch.cam_layer = "model_4.stages.3"
        arch.embedding_layer = "model_4.head.flatten"
        arch.classifier_layer = "model_4.head.fc"

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

    def freeze_feature_extractor(self):
        # the hgmodels all have .fc as the final classification layer
        # so we can freeze all except each model's .fc
        return self.freeze_layers_except([m.fc for m in self.network.models])

    def change_classes(*args, **kwargs):
        raise NotImplementedError(
            """HawkEars contains an ensemble of models. Think carefully about
            what you want to do: perhaps you want to create a separate
            classifier which you can train on embeddings from HawkEars: 

            ```python
            from opensoundscape.ml import shallow_classifier
            hawkears = HawkEars()
            clf = shallow_classifier.MLPClassifier(2048,n_classes,())
            shallow_classifier.fit_classifier_on_embeddings(embedding_model=hawkears, classifier_model=clf, ...)
            """
        )

    @classmethod
    def load(cls, path):
        """reload object after saving to file with .save()

        Args:
            path: path to file saved using .save()

        Returns:
            new HawkEars instance

        Note: Note that if you used pickle=True when saving, the model object might not load properly
        across different versions of OpenSoundscape.
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
            model = cls()
            # load up the weights and instantiate from dictionary keys
            # includes preprocessing parameters and settings
            state_dict = loaded_content.pop("weights")

            # load weights from checkpoint
            model.network.load_state_dict(state_dict)
        else:
            model = loaded_content  # entire pickled object, not dictionary

        return model
