import platform
import pandas as pd
import numpy as np
import torch
import warnings
import numpy as np
from pathlib import Path

from bioacoustics_model_zoo.utils import (
    AudioSampleArrayDataloader,
    register_bmz_model,
)

from bioacoustics_model_zoo.utils import register_bmz_model
from bioacoustics_model_zoo.tensorflow_wrapper import (
    TensorFlowModelWithPytorchClassifier,
)

import torch


@register_bmz_model
class MultiSpeciesWhale(TensorFlowModelWithPytorchClassifier):
    """load the multi-species hale bioacoustic classifier from Kaggle

    See model card and attributions at: 
    https://www.kaggle.com/models/google/multispecies-whale/TensorFlow2/default/1

    Takes 5s audio windows at 24 kHz, performs multi-target classification outputs on 11 classes; embedding shape is 1280 (EfficientNet B0)\
        
    Terms of Use
    This model has been developed as part of the AI for Nature and Society program
    at Google. The developers request that users adhere to Google’s AI principles,
    in particular #1 “Be socially beneficial." in only pursuing applications which
    have societal and/or environmental benefit, as well as wildlife conservation for
    not-for-profit decision-making, education, or research. (The official license
    remains Apache 2.0.) If you have any questions about appropriate use cases for
    this model, please contact bioacoustics-project@google.com.

    Class Common Name :     Class Code
    Humpback	            Mn
    Orca	                Oo  
    Bryde's	                Be
    Minke	                Ba
    Blue	                Bm
    Fin	                    Bp
    Right (Atlantic)	    Eg
    Right (Pacific, upcall)	Upcall
    Right (Pacific, gunshot)Gunshot
    Orca echolocation	    Echolocation
    Orca whistle	        Whistle
    Orca call	            Call

    > Developer Note: For echolocation, whistle, and call, we do not expect high
    orca specificity. Nevertheless, we qualified the names with orca because a
    strong majority of our training data came from that species.


    Args:
        version: select from released versions on Kaggle [Default: 1]
        device: selects GPU vs CPU for the tensorflow model [Default: None]

    Methods:
        predict: get per-audio-clip per-class scores as pandas DataFrame
        embed: generate embedding layer outputs for samples
        forward: return selected outputs as a dictionary with keys: 'logit', 'feature', 'spectrogram', 'custom_classifier'

    Example Usage:
    ```
    import bioacoustics_model_zoo as bmz
    model=bmz.MultiSpeciesWhale()
    predictions = model.predict(['test.wav'], clip_step=1.0) # generate logit scores for 3.91s audio windows with 1s step size
    model.predict(file, clip_step=1.0, batch_size=32, activation_layer='sigmoid') # 0-1 output scores and >1 batch size (use large batch size for GPUs)
    embeddings = model.embed(['test.wav']) #generate 2048-dimensional embeddings on audio windows
    all_outputs = model.forward(['test.wav']) #get all model outputs including spectrograms and spatial embeddings
    all_outputs['feature'].shape, all_outputs['logit'].shape, all_outputs['spectrogram'].shape
    ```

    Environment setup:
    MultiSpeciesWhale requires tensorflow and kagglehub packages, which can be installed with
    ```
    pip install --upgrade opensoundscape bioacoustics-model-zoo tensorflow kagglehub
    ```

    Note: because TensorFlow Hub implements its own caching system, we do not use the bioacoustics
    model zoo caching functionality here. TF Hub caches to a temporary directory by default (does not
    persist across system restart), but this can be configured
    (see https://www.tensorflow.org/hub/caching#caching_of_compressed_downloads)

    """

    def __init__(self, version=1, device=None, use_common_names=False):
        """initialize MultiSpeciesWhale BMZ model from TensorFlow Hub

        Args:
            version: select from released versions of Google Multispecies-Whale model on Kaggle
            device: selects GPU vs CPU for the tensorflow model
                Note that different models are downloaded from TF Hub for GPU vs CPU usage.
                - default [None]: uses GPU if available, otherwise CPU
                - 'cpu': forces CPU usage
                - 'cuda': forces GPU usage
            use_common_names: if True, uses common names rather than class codes
                (see self.class_dict for mapping of class codes to common names)

        """
        # only require tensorflow and kagglehub if/when this class is used
        try:
            import tensorflow as tf
            import kagglehub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                """MultiSpeciesWhale requires tensorflow and kagglehub packages >=2.20.0.
                Please install them using:
                pip install --upgrade opensoundscape bioacoustics-model-zoo tensorflow kagglehub
                """
            ) from exc

        # which model to load depends on whether GPU is available
        if device is None:
            if torch.cuda.is_available():
                self.tf_device = tf.device("GPU")
                self.device = "cuda"
            else:
                self.tf_device = tf.device("CPU")
                self.device = "cpu"
        else:
            self.device = device

        tested_versions = (1,)  # as of September 2026
        handle = f"google/multispecies-whale/TensorFlow2/default/{version}"

        if not version in tested_versions:
            warnings.warn(
                f"version {version} has not been tested on {device}, tested versions: {tested_versions}"
            )

        # tensorflow tends to choose the device automatically, so to manually select between CPU and GPU we need to use the tf.device
        # context manager both when the model is loaded and when the model forward call is made
        with self.tf_device:
            try:
                model_path = kagglehub.model_download(handle)
                tf_model = tf.saved_model.load(Path(model_path) / "multispecies_whale/")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from KaggleHub at {handle}. "
                ) from e

        metadata_fn = tf_model.signatures["metadata"]
        classes = metadata_fn()["class_names"].numpy().astype(str).tolist()
        sr = metadata_fn()["input_sample_rate"].numpy()  # 24 kHz
        sample_duration = metadata_fn()["context_width_samples"].numpy() / sr  # 5 s
        # initialize parent class with methods for training custom classifier head
        super().__init__(
            embedding_size=1280,
            classes=classes,
            sample_duration=sample_duration,
            sample_rate=sr,
        )
        self.class_dict = {
            "Oo": "Orca",
            "Mn": "Humpback",
            "Eg": "Right (Atlantic)",
            "Be": "Bryde's",
            "Upcall": "Right (Pacific, upcall)",
            "Bp": "Fin",
            "Call": "Orca call",
            "Gunshot": "Right (Pacific, gunshot)",
            "Echolocation": "Orca echolocation",
            "Bm": "Blue",
            "Whistle": "Orca whistle",
            "Ba": "Minke",
        }
        # ensure same order
        self.common_names = [self.class_dict[c] for c in self.classes]
        self.class_codes = self.classes.copy()
        if use_common_names:
            self.classes = [self.class_dict[c] for c in self.classes]

        # store version number as attribute
        self.name = "google-multispecies-whale"
        self.version = version
        self.tf_model = tf_model
        self.inference_dataloader_cls = AudioSampleArrayDataloader
        self.train_dataloader_cls = AudioSampleArrayDataloader

        # preprocessing notes:
        # no specific resampling algorithm is suggested in the Kaggle usage page
        # during inference, audio could optionally be scaled, but the default is no scaling

        # if on a mac, disable XLA JIT to avoid TF hanging behavior (as of TF 2.21.0, March 2026)
        if platform.system() == "Darwin":
            warnings.warn(
                "Disabling TensorFlow's XLA compilation (setting tf.config.optimizer.set_jit(False)) because otherwise "
                "TF models on Mac hang at runtime as of Tensorflow 2.21.0"
            )
            tf.config.optimizer.set_jit(False)

    def batch_forward(
        self,
        batch_samples,
        targets=("logit", "feature", "spectrogram"),
        avgpool=False,
    ):
        """run inference on a single batch of samples

        Returns a dictionary of outputs for the each target

        Args:
            batch_data: np.array of audio samples, shape (batch_size, 32000*5)
            targets: tuple of str, select from
                ['embedding', 'spatial_embedding', 'labels', 'spectrogram','custom_classifier']
                - 'custom_classifier' is the result of self.network() on the embeddings
            avgpool: ignored
        Returns:
            dict with keys matching targets, values are np.arrays of outputs
        """
        import tensorflow as tf

        waveform = tf.convert_to_tensor(
            np.array([s.data.samples for s in batch_samples], dtype=np.float32),
            dtype=tf.float32,
        )
        waveform = tf.expand_dims(waveform, axis=-1)  # add channel dimension

        # call model in context manager so it actually uses the CPU (even if a GPU is available)
        # we already loaded a single inference clip in each sample; no internal windowing
        # w = tf.cast(1e12, tf.int64)
        with self.tf_device:
            model_outputs = {}

            spec = self.tf_model.front_end(waveform)[:, :128, :]
            model_outputs["spectrogram"] = spec
            if "logit" in targets or -1 in targets:
                model_outputs["logit"] = self.tf_model.logits(spec)
            if "feature" in targets or "custom_classifier" in targets:
                model_outputs["feature"] = self.tf_model.features(spec)
                if "custom_classifier" in targets or (
                    -1 in targets and self.use_custom_classifier
                ):
                    emb_tensor = torch.tensor(model_outputs["feature"]).to(self.device)
                    self.network.to(self.device)
                    custom_classifier = self.network(emb_tensor).detach().cpu()
                    model_outputs["custom_classifier"] = custom_classifier

        # opensoundscape uses reserved key -1 for model outputs e.g. during .predict()
        if -1 in targets:
            if self.use_custom_classifier:
                model_outputs[-1] = model_outputs["custom_classifier"]
            else:
                model_outputs[-1] = model_outputs["logit"]

        # only retaining requested outputs
        model_outputs = {
            k: None if v is None else v.numpy()
            for k, v in model_outputs.items()
            if k in targets
        }

        return model_outputs

    def forward(
        self,
        samples,
        progress_bar=True,
        wandb_session=None,
        targets=("logit", "feature", "spectrogram"),
        return_dfs=True,
        clip_step=1.0,
        **dataloader_kwargs,
    ):
        """
        Run inference on a list of samples, returning all selected outputs as a dictionary

        use "custom_classifier" in return_values to get outputs from the
        custom classifier head (self.network)

        wraps self.predict_dataloader() and self.__call__() to run the model on
        audio files/clips, then optionally places 1D outputs in dataframes

        Args:
            samples: list of file paths, OR pd.DataFrame with index containing audio file paths
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            wandb_session: wandb.Session object, if provided, logs progress
            targets: tuple(str,): select from 'logit', 'feature', 'spectrogram', 'custom_classifier'
                [default: ('logit','feature','spectrogram')]
                Include any combination of the following:
                - 'logit': logit scores (class predictions) on the species classes
                - 'feature': 1D feature vectors from the penultimate layer of the network
                - 'spatial_embedding': un-pooled spatial embeddings from the network
                - 'spectrogram': log-mel spectrograms generated during preprocessing
                - 'custom_classifier': outputs of the custom classifier head (self.network)
            return_dfs: bool, if True, returns outputs as pd.DataFrame with multi-index like
                .predict() ('file','start_time','end_time'), if False, returns np.array
                [default: True]
            clip_step: float, step size in seconds between subsequent inference windows
                - default of 1.0 matches the hard-coded values in Kaggle examples
            **dataloader_kwargs: additional keyword arguments passed to the dataloader such
                as batch_size, num_workers, etc.

        Returns: dictionary with content depending on return_values and return_dfs arguments:
            - 'label': pd.DataFrame or np.array of per-clip logits on species classes
                shape: (num_clips, num_classes)
            - 'feature': pd.DataFrame or np.array of per-clip 1D feature vectors
                shape: (num_clips, 1536)
            - 'spatial_embedding': np.array of per-clip spatial embeddings
                shape: (num_clips, 5, 3, 1536)
            - 'custom_classifier': pd.DataFrame or np.array of per-clip logits from the
                custom classifier head, shape: (num_clips, num_custom_classes)
        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(
            samples, clip_step=clip_step, **dataloader_kwargs
        )

        # run inference, getting all outputs
        # avoids aggregating unrequested outputs to save memory
        results_dict = self(
            dataloader=dataloader,
            wandb_session=wandb_session,
            progress_bar=progress_bar,
            targets=targets,
        )

        # optionally put 1D outputs in DataFrames with multi-index ('file','start_time','end_time')
        # and appropriate column names
        if return_dfs:
            if "logit" in results_dict:
                results_dict["logit"] = pd.DataFrame(
                    data=results_dict["logit"],
                    index=dataloader.dataset.dataset.label_df.index,
                    columns=self._original_classes,
                )
            if "feature" in results_dict:
                results_dict["feature"] = pd.DataFrame(
                    data=results_dict["feature"],
                    index=dataloader.dataset.dataset.label_df.index,
                    columns=None,
                )
            if "custom_classifier" in results_dict:
                results_dict["custom_classifier"] = pd.DataFrame(
                    data=results_dict["custom_classifier"],
                    index=dataloader.dataset.dataset.label_df.index,
                    columns=self._custom_classes,
                )

        return results_dict

    def _check_or_get_default_embedding_layer(self, target_layer=None):
        """always "feature" for this model"""
        return "feature"
