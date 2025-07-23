# Base configuration. Specific configurations only have to specify the parameters they're changing.
# this file is copied directly from https://github.com/jhuus/HawkEars/blob/0.1.0/core/base_config.py
# and corresponds to tag 0.1.0
from dataclasses import dataclass


@dataclass
class Audio:
    # sampling rate should be a multiple of spec_width / segment_len,
    # so that hop length formula gives an integer (segment_len * sampling_rate / spec_width)

    spec_height = 192  # spectrogram height
    spec_width = 384  # spectrogram width (3 * 128)
    win_length = 2048
    min_audio_freq = 200  # need this low for American Bittern
    max_audio_freq = 13000  # need this high for Chestnut-backed Chickadee "seet series"
    mel_scale = True
    
    segment_len = 3  # spectrogram duration in seconds
    sampling_rate = 37120
    choose_channel = True  # use heuristic to pick the cleanest audio channel
    check_seconds = 3  # check segment of this length when picking cleanest channel
    power = 1.0
    spec_block_seconds = (
        240  # max seconds of spectrogram to create at a time (limited by GPU memory)
    )

    # low-frequency audio settings for Ruffed Grouse drumming identifier
    low_band_spec_height = 64
    low_band_win_length = 4096
    low_band_min_audio_freq = 30
    low_band_max_audio_freq = 200
    low_band_mel_scale = False


@dataclass
class BaseConfig:
    audio = Audio()
