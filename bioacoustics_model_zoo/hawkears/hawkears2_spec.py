# import torch
# import torchaudio
# import numpy as np
# import torch.nn.functional as F

# from opensoundscape.preprocess.actions import BaseAction
# from opensoundscape.preprocess.actions import register_action_cls

# from bioacoustics_model_zoo.hawkears import hawkears_base_config


# @register_action_cls
# class HawkEarsSpec(BaseAction):
#     """hawkears preprocessing of audio signal to normalized spectrogram

#     uses settings from config file's BaseConfig class

#     Args:
#         cfg: if None, loads BaseConfig from hawkears_base_config module
#             - can be a config object from hawkears repo
#         low_band: if True, creates low frequency spectrograms for a specialized model
#             otherwise creates typical spectrograms for bird classification
#         device: torch device (or string name) to use for spectrogram creation
#             - eg, 'mps', 'cuda:0', 'cpu'
#             - as of April 2024, torchaudio supports cuda but not mps for making spectrograms
#             - default 'cpu' is safest best but slowest

#     based on:
#     https://github.com/jhuus/HawkEars1/blob/24bc5a3e031866bc3ff81343bffff83429ee7897/core/audio.py
#     """

#     def __init__(self, cfg=None, low_band=False, device="cpu"):
#         super(HawkEarsSpec, self).__init__()
#         self.low_band = low_band

#         if low_band:  # special settings if we want to detect Ruffed Grouse
#             min_audio_freq = self.cfg.audio.low_band_min_audio_freq
#             max_audio_freq = self.cfg.audio.low_band_max_audio_freq
#             spec_height = self.cfg.audio.low_band_spec_height
#             # mel_scale = self.cfg.audio.low_band_mel_scale
#             self.freq_scale = self.cfg.audio.low_band_scale
#         else:
#             min_audio_freq = self.cfg.audio.min_audio_freq
#             max_audio_freq = self.cfg.audio.max_audio_freq
#             spec_height = self.cfg.audio.spec_height
#             # mel_scale = self.cfg.audio.mel_scale
#             self.freq_scale = self.cfg.audio.scale

#         # use custom config if provided, otherwise default
#         if cfg is None:
#             cfg = hawkears_base_config.BaseConfig()
#         self.cfg = cfg

#         # set device (mps/cuda/cpu) to use for spectrogram creation
#         self.device = torch.device(device)

#         fft_hop = int(
#             cfg.audio.segment_len * cfg.audio.sampling_rate / cfg.audio.spec_width
#         )

#         self.linear_transform = torchaudio.transforms.Spectrogram(
#             n_fft=2 * self.cfg.audio.win_length,
#             win_length=self.cfg.audio.win_length,
#             hop_length=fft_hop,
#             power=self.cfg.audio.power,
#         )
#         self.linear_transform.to(self.device)

#         self.mel_transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=self.cfg.audio.sampling_rate,
#             n_fft=2 * self.cfg.audio.win_length,
#             win_length=self.cfg.audio.win_length,
#             hop_length=fft_hop,
#             f_min=min_audio_freq,
#             f_max=max_audio_freq,
#             n_mels=spec_height,
#             power=self.cfg.audio.power,
#         )
#         self.mel_transform.to(self.device)

#     def _normalize(self, spec):
#         """normalize values to have range 0-1"""
#         spec -= spec.min()
#         max = spec.max()
#         if max > 0:
#             spec = spec / max
#         return spec.clip(0, 1)

#     def _get_raw_spectrogram(self, signal, low_band=False):
#         """use config settings to create linear or mel spectrogram"""

#         signal = signal.reshape((1, signal.shape[0]))
#         tensor = torch.from_numpy(signal).to(self.device)

#         if self.freq_scale == "log":
#             spec = self.linear_transform(tensor)  # [1, n_freqs, n_frames]
#             spec = torch.matmul(
#                 self.log2_filterbank, spec.squeeze(0)
#             )  # [n_mels, n_frames]
#             spec = spec.unsqueeze(0)  # [1, n_mels, n_frames]
#         elif self.freq_scale == "mel":
#             spec = self.mel_transform(tensor)  # [1, n_mels, T]
#         elif self.freq_scale == "linear":
#             spec = self.linear_transform(tensor)
#             freqs = torch.fft.rfftfreq(
#                 2 * self.win_length, d=1 / self.cfg.audio.sampling_rate
#             )
#             mask = (freqs >= self.cfg.audio.min_freq) & (
#                 freqs <= self.cfg.audio.max_freq
#             )
#             spec = spec[:, mask, :].unsqueeze(1)  # [1, 1, F_sel, T]

#             # downsample frequency to spec_height (energy-preserving)
#             spec = F.interpolate(
#                 spec,
#                 size=(self.cfg.audio.spec_height, spec.shape[-1]),
#                 mode="area",
#             )
#             spec = spec.squeeze(1)  # [1, F, T]

#         if self.cfg.audio.decibels:
#             # Apply dB conversion on CPU (already numpy)
#             spec = torch.from_numpy(spec).unsqueeze(0)  # [1, F, T]
#             spec = torchaudio.transforms.AmplitudeToDB(
#                 stype="power", top_db=self.cfg.audio.top_db
#             )(spec)
#             spec = spec**self.cfg.audio.db_power
#             spec = spec[0].numpy()

#         return spec

#     def __call__(self, sample):
#         """creates spectrogram, normalizes, casts to torch.tensor"""
#         # sample.data will be Audio object. Replace sample.data with torch.tensor of spectrogram.
#         spec = self._get_raw_spectrogram(sample.data.samples, low_band=self.low_band)

#         # normalize
#         spec = self._normalize(spec)

#         # reshape if needed (https://github.com/jhuus/HawkEars1/blob/f924114ebe6e6f220df74f9fb136f6194f7ac0e8/core/audio.py#L150C17-L153C17)
#         spec = spec[: self.cfg.audio.spec_height, : self.cfg.audio.spec_width]
#         if spec.shape[1] < self.cfg.audio.spec_width:
#             spec = np.pad(
#                 spec,
#                 ((0, 0), (0, self.cfg.audio.spec_width - spec.shape[1])),
#                 "constant",
#                 constant_values=0,
#             )

#         # update the AudioSample's .data in-place
#         sample.data = torch.tensor(spec).unsqueeze(0)
