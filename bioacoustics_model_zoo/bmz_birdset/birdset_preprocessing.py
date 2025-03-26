"""Implement opensoudnscape preprocessor for BirdSet models

Adapted from https://github.com/DBD-research-group/BirdSet/blob/guide-model-inference/birdset/modules/models/birdset_models/convnext_bs.py on 2025-03-20

Individual models have subclassed preprocessors with hard-coded parameters, e.g. ConvNextBirdsetPreprocessor
"""

import torch
import torchaudio
import pandas as pd

from opensoundscape.preprocess import action_functions
from opensoundscape.preprocess.actions import (
    Action,
    AudioClipLoader,
    AudioTrim,
    register_action_fn,
)
from opensoundscape.preprocess.overlay import Overlay


class PowerToDB(torch.nn.Module):
    """
    A power spectrogram to decibel conversion layer. See birdset.datamodule.components.augmentations
    """

    def __init__(self, ref=1.0, amin=1e-10, top_db=80.0):
        super(PowerToDB, self).__init__()
        # Initialize parameters
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

    def forward(self, S):
        # Convert S to a PyTorch tensor if it is not already
        S = torch.as_tensor(S, dtype=torch.float32)

        if self.amin <= 0:
            raise ValueError("amin must be strictly positive")

        if torch.is_complex(S):
            magnitude = S.abs()
        else:
            magnitude = S

        # Check if ref is a callable function or a scalar
        if callable(self.ref):
            ref_value = self.ref(magnitude)
        else:
            ref_value = torch.abs(torch.tensor(self.ref, dtype=S.dtype))

        # Compute the log spectrogram
        log_spec = 10.0 * torch.log10(
            torch.maximum(magnitude, torch.tensor(self.amin, device=magnitude.device))
        )
        log_spec -= 10.0 * torch.log10(
            torch.maximum(ref_value, torch.tensor(self.amin, device=magnitude.device))
        )

        # Apply top_db threshold if necessary
        if self.top_db is not None:
            if self.top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = torch.maximum(log_spec, log_spec.max() - self.top_db)

        return log_spec


from opensoundscape.preprocess.preprocessors import (
    BasePreprocessor,
    register_preprocessor_cls,
)


@register_action_fn
def torchaudio_spectrogram(audio, n_fft=1024, hop_length=320, power=2.0, **kwargs):
    """
    Create a spectrogram from an audio signal using PyTorch.

    Takes opensoundscape.Audio, returns pytorch tensor

    Args:
        n_fft (int): Number of FFT points.
        hop_length (int): Number of samples between frames.
        power (float): Power to raise the magnitude to.
    Returns:
        function: A function that takes an audio tensor and returns its spectrogram.
    """
    # unsqueeze adds leading channel dim
    return torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length, power=power, **kwargs
    )(audio).unsqueeze(0)


@register_action_fn
def spec_to_melspec(spec_tensor, **kwargs):
    return torchaudio.transforms.MelScale(**kwargs)(spec_tensor)


@register_action_fn
def birdset_powertodb(melspec, top_db=80, ref=1.0, amin=1e-10, **kwargs):
    """
    Convert a mel spectrogram to decibels using the PowerToDB class.
    """
    return PowerToDB(ref=ref, amin=amin, top_db=top_db)(melspec)


@register_action_fn
def torch_resample(audio, sample_rate=32000):
    """
    Resample the audio signal to a new sample rate using PyTorch.

    Args:
        audio (opensoundscape.Audio): The audio object to be resampled.
        sample_rate (int): The target sample rate.
    Returns:
        torch.Tensor: The resampled audio tensor.
    """
    return torchaudio.transforms.Resample(
        orig_freq=audio.sample_rate, new_freq=sample_rate
    )(torch.tensor(audio.samples))


@register_preprocessor_cls
class BirdsetPreprocessor(BasePreprocessor):
    def __init__(self, sample_duration=5, sample_rate=32000, overlay_df=None):
        super().__init__(sample_duration=sample_duration)

        self.pipeline = pd.Series(
            {
                "load_audio": AudioClipLoader(sample_rate=None),
                # if we are augmenting and get a long file, take a random trim from it
                "random_trim_audio": AudioTrim(
                    target_duration=sample_duration,
                    is_augmentation=True,
                    random_trim=True,
                ),
                # otherwise, we expect to get the correct duration. no random trim
                # trim or extend (w/silence) clips to correct length
                "trim_audio": AudioTrim(
                    target_duration=sample_duration, random_trim=False
                ),
                "resample": Action(torch_resample, sample_rate=sample_rate),
                "to_spec": Action(
                    torchaudio_spectrogram,
                    is_augmentation=False,
                ),
                "to_mel": Action(
                    spec_to_melspec,
                    is_augmentation=False,
                ),
                "power_to_db": Action(
                    birdset_powertodb,
                    is_augmentation=False,
                ),
                # Overlay is a version of "mixup" that draws samples from a user-specified dataframe
                # and overlays them on the current sample
                "overlay": (
                    Overlay(
                        is_augmentation=True,
                        overlay_df=pd.DataFrame() if overlay_df is None else overlay_df,
                        update_labels=True,
                    )
                ),
                # add vertical (time) and horizontal (frequency) masking bars
                "time_mask": Action(action_functions.time_mask, is_augmentation=True),
                "frequency_mask": Action(
                    action_functions.frequency_mask, is_augmentation=True
                ),
                # add noise to the sample
                "add_noise": Action(
                    action_functions.tensor_add_noise, is_augmentation=True, std=0.005
                ),
                # linearly scale the _values_ of the sample based on a predefined mean and std
                # BirdSet preprocessing uses AudioSet mean/std
                "rescale": Action(
                    action_functions.scale_tensor,
                    input_mean=-4.268,
                    input_std=4.569,
                ),
                # apply random affine (rotation, translation, scaling, shearing) augmentation
                # default values are reasonable for spectrograms: no shearing or rotation
                "random_affine": Action(
                    action_functions.torch_random_affine, is_augmentation=True
                ),
            }
        )

        # bypass overlay if overlay_df was not provided (None)
        # keep the action in the pipeline for ease of enabling it later
        if overlay_df is None or len(overlay_df) < 1:
            self.pipeline["overlay"].bypass = True
