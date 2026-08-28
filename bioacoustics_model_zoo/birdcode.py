from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np
import torch
import birdnames
from opensoundscape.ml.cnn import SpectrogramClassifier, register_model_cls
from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from sound_event_detection.models import FrameDetector

from bioacoustics_model_zoo.utils import register_bmz_model


@register_bmz_model
@register_model_cls
class BirdCODE(SpectrogramClassifier):
    def __init__(self, version=None, device=None, class_names="scientific"):
        """Initialize the Earth Species project BirdCODE pytorch CNN from huggingface

        Implementation Notes:
        - internally applies sigmoid to forward pass logits
        - when pooling frame scores to clips, uses tempered pooling with t=1.0 (=linear softmax pooling)

        Required package installation:
        This model uses earthspecies/sound-event-detection, which is not currently on PyPi. Install directly
        from github with:
        `pip install git+https://github.com/earthspecies/sound-event-detection.git`
        or follow the repo instructions for installation with uv.

        Args:
            version: str, optional, git revision to load. If None, loads the latest version.
                - Note: as of August 2026, there are no available pinned versions, only 'main'
                pointing to latest version
            device: str, optional, device to run the model on. One of 'cpu' or 'cuda' ('mps'
                implementation is currently broken).
            class_names: str, optional, which class names to use for the output.
                One of 'scientific', 'common', 'alpha', or 'ebird'.
                - if not 'scientific', uses the `birdnames` package to convert class names, but note
                that missing class names will become `None`!
        """
        if device == "mps":
            raise ValueError(
                "MPS inference is not currently working correctly (https://github.com/earthspecies/sound-event-detection/issues/3). Use cpu or cuda."
            )
        network = (
            FrameDetector.from_hf_hub(
                "EarthSpeciesProject/sed-birdcode", revision=version
            )
            .eval()
            .to(device)
        )
        if class_names == "scientific":
            classes = network.labels
        elif class_names == "common":
            classes = birdnames.common(network.labels)
        elif class_names == "alpha":
            classes = birdnames.alpha(network.labels)
        elif class_names == "ebird":
            classes = birdnames.ebird(network.labels)
        else:
            raise ValueError(
                f"Invalid class_names: {class_names}. Must be one of 'scientific', 'common', 'alpha', or 'ebird'."
            )

        super().__init__(
            architecture=network,
            classes=classes,
            sample_duration=network.window_duration,
            sample_rate=32000,
            preprocessor_cls=AudioAugmentationPreprocessor,
            device=device,
        )
        self.frame_rate = network.frame_rate
        self.frame_duration = 1 / self.frame_rate
        self._frame_start_ts = np.arange(0, self.sample_duration, self.frame_duration)
        self._frames_per_window = len(self._frame_start_ts)
        self.network = network
        self.network.embedding_layer = self.network.encoder

    @property
    def classifier(self):
        """Return the underlying FrameDetector.classifier"""
        return self.network.classifier

    @property
    def embedding_dim(self):
        return self.network.classifier.in_features

    def batch_forward(self, batch, targets=None, avgpool=True):
        """Generate clip-level predictions and/or clip or frame-level embeddings for a batch of AudioSample objects.

        Args:
            batch: list of opensoundscape AudioSample objects (typically produced by a dataloader class, eg object created by self.predict_dataloader())
            targets: list of targets to return; can include self.class_outputs_key for logits and/or self.network.encoder to get embeddings
            avgpool: bool, if True, average frame embeddings across time dimension to produce a single embedding per clip; if False, return frame embeddings for each frame in the clip

        Returns:
            - dict with keys 'class_outputs' and/or 'frame_embeddings', depending on targets
            - when returning embeddings, shape is (batch, frames=38, embedding_dim) if avgpool=False, or (batch, embedding_dim) if avgpool=True
        """
        # TODO: this seems to include unnecessary round-trip casting torch-numpy-torch
        batch_audio = np.array([s.data.samples for s in batch])
        outs = {}

        if "encoder" in targets:
            outs["encoder"] = (
                self.network.encoder(
                    torch.as_tensor(batch_audio, device=self.device),
                )
                .detach()
                .cpu()
            )
            if avgpool:  # average across dim frames within clip
                outs["encoder"] = outs["encoder"].mean(1)

        # if returning both scores and embeddings, this code currently runs forward pass twice
        # However, I wasn't able to put a forward hook on the encoder to get the embeddings
        # because it is called with encoder.forward() rather than encoder()
        if self.class_outputs_key in targets:
            outs[self.class_outputs_key] = self.network.run_as_classifier(
                batch_audio,
                overlap=0,
                device=self.device,
                batch_size=len(batch_audio),
            ).predictions

        return outs

    def predict_frames(
        self, samples, batch_size=1, overlap_fraction=None, return_df=True
    ):
        """Predicts frame-level predictions for a single audio file or a list of audio files.

        Warning: embedding dimension is very large (188,440)

        Args:
           samples: audio file or list of files
           batch_size: int, optional, number of files to simultaneously process in forward pass
           overlap_fraction: float, optional, % overlap between windows

        Returns:
        if return_df is True:
            predictions: dataframe, shape (num_frames, num_classes)
                - columns are class names, rows are frames
                - multi-index (file, start_time, end_time) for each frame
        if return_df is False:
            predictions: np.ndarrays, (frames, classes) across all files
        """
        loader = self.predict_dataloader(
            samples,
            batch_size=batch_size,
            overlap_fraction=0,  # overlap is handled by network.run()
        )
        all_preds = []
        all_files = []
        all_frame_starts = []
        for batch in tqdm(loader):
            bs = len(batch)
            batch_audio = np.array([s.data.samples for s in batch])
            batch_frame_starts = [
                t + s.start_time for s in batch for t in self._frame_start_ts
            ]
            batch_files = [
                s.source for s in batch for _ in range(self._frames_per_window)
            ]
            frame_preds = np.vstack(
                self.network.run(
                    batch_audio,
                    batch_size=bs,
                    overlap=overlap_fraction,
                    device=self.device,
                ).predictions
            )
            all_preds.append(frame_preds)
            all_files.append(batch_files)
            all_frame_starts.append(np.hstack(batch_frame_starts))
        all_preds = np.vstack(all_preds)
        all_files = np.hstack(all_files)
        all_frame_starts = np.hstack(all_frame_starts)

        if return_df:
            return pd.DataFrame(
                np.vstack(all_preds),
                index=pd.MultiIndex.from_arrays(
                    [
                        all_files,
                        all_frame_starts,
                        all_frame_starts + self.frame_duration,
                    ],
                    names=["file", "start_time", "end_time"],
                ),
                columns=self.classes,
            )
        else:
            return np.vstack(all_preds)

    def _check_or_get_default_embedding_layer(self, target_layer=None):
        """Check if target_layer is valid, or return default embedding layer (self.network.encoder)"""
        if target_layer in (None, self.network.encoder, "encoder"):
            return "encoder"  # refers to self.network.encoder
        else:  # attempt to use user-specified layer
            return target_layer

    def generate_cams(self, *args, **kwargs):
        raise NotImplementedError(
            "Class activation mapping is not currently implemented for BirdCODE, as spectrogram creation happens internally."
        )
