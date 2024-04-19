from opensoundscape import Audio, CNN
from opensoundscape.ml.cnn import BaseClassifier
import pandas as pd
import cv2

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.preprocess.actions import Action, BaseAction

from bioacoustics_model_zoo.hawkears import hawkears_base_config

# Define custom EfficientNet_v2 configurations.

from timm.models import efficientnet


def build_efficientnet_architecture(model_name, **kwargs):
    if model_name == "1":
        # ~ 1.5M parameters
        channel_multiplier = 0.4
        depth_multiplier = 0.4
    elif model_name == "2":
        # ~ 2.0M parameters
        channel_multiplier = 0.4
        depth_multiplier = 0.5
    elif model_name == "3":
        # ~ 3.4M parameters
        channel_multiplier = 0.5
        depth_multiplier = 0.6
    elif model_name == "4":
        # ~ 4.8M parameters
        channel_multiplier = 0.6
        depth_multiplier = 0.6
    elif model_name == "5":
        # ~ 5.7M parameters
        channel_multiplier = 0.6
        depth_multiplier = 0.7
    elif model_name == "6":
        # ~ 7.5M parameters
        channel_multiplier = 0.7
        depth_multiplier = 0.7
    elif model_name == "7":
        # ~ 8.3M parameters
        channel_multiplier = 0.7
        depth_multiplier = 0.8
    else:
        raise Exception(f"Unknown custom EfficientNetV2 model name: {model_name}")

    return efficientnet._gen_efficientnetv2_s(
        "efficientnetv2_rw_t",
        channel_multiplier=channel_multiplier,
        depth_multiplier=depth_multiplier,
        in_chans=1,
        **kwargs,
    )


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

    def go(self, sample):
        """creates spectrogram, normalizes, casts to torch.tensor"""
        # sample.data will be Audio object. Replace sample.data with torch.tensor of spectrogram.
        spec = self._get_raw_spectrogram(sample.data.samples)

        # normalize
        spec = self._normalize(spec)

        # update the AudioSample's .data in-place
        sample.data = torch.tensor(spec).unsqueeze(0)


import torch
from bioacoustics_model_zoo.utils import download_github_file
from pathlib import Path


class HawkEars(CNN):
    """HawkEars bird classification CNN for 314 North American sp

    Args:
        config: use None for default, or pass a valid object created with the HawkEars repo
        checkpoint_url: URL or local file path to model checkpoint. Default is recommended.
            Note that some values in this class (such as the architecture) are hard coded to match
            the parameters of the default checkpoint model.

    Usage: use .predict([audio1.wav,audio2.wav]) to generate class predictions on audio files. See OpenSoundscape.CNN for details.
    """

    ##TODO: implement embed (maybe in CNN rather than here)

    def __init__(
        self,
        config=None,
        checkpoint_url="https://github.com/jhuus/HawkEars/blob/24bc5a3e031866bc3ff81343bffff83429ee7897/data/ckpt/custom_efficientnet_5B.ckpt",
    ):
        # download the necessary files:
        # download model if URL, otherwise find it at local path:
        if checkpoint_url.startswith("http"):
            print("downloading model from URL...")
            model_path = download_github_file(checkpoint_url)
        else:
            model_path = checkpoint_url
        model_path = str(Path(model_path).resolve())  # get absolute path as string
        assert Path(model_path).exists(), f"Model path {model_path} does not exist"

        # load list of classes
        class_list_path = Path(__file__).with_name("classes_edited.txt")
        classes = pd.read_csv(class_list_path)  # two columns: common, alpha

        # this checkpoint was trained on 314 classes
        arch = build_efficientnet_architecture("5", num_classes=314)
        model_dict = torch.load(model_path, map_location="cpu")

        # remove 'base_model' prefix from state dict keys to match our architecture
        state_dict = {
            k.replace("base_model.", ""): v for k, v in model_dict["state_dict"].items()
        }
        arch.load_state_dict(state_dict)

        # initialize the CNN object with this architecture and class list
        # use 3s duration and expected sample shape for HawkEars
        super(HawkEars, self).__init__(
            arch,
            classes=classes["Common Name"],
            sample_duration=3,
            sample_shape=[192, 384, 1],
        )

        # compose the preprocessing pipeline:
        # load audio with 3s framing; extend to 3s if needed
        # create spectrogram based on config values, using same functions as HawkEars to ensure same results
        # normalize spectrogram to max=1
        pre = AudioPreprocessor(sample_duration=3, sample_rate=40960)
        pre.insert_action(
            action_index="extend",
            action=Action(Audio.extend_to, is_augmentation=False, duration=3),
        )

        # note: can specify pre.pipeline.to_spec.device = 'cuda' to use cuda for spectrogram creation
        # but mps does not support spectrogram creation as of April 2024
        pre.insert_action(action_index="to_spec", action=HawkEarsSpec(cfg=config))
        self.preprocessor = pre
