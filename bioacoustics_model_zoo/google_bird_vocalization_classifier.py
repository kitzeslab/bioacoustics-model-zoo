from pathlib import Path

import torch
import pandas as pd
import numpy as np

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier


from bioacoustics_model_zoo.utils import (
    collate_to_np_array,
    AudioSampleArrayDataloader,
)


# TODO: update url to v3 when its no longer broken
def google_bird_vocalization_classifier(
    url="https://tfhub.dev/google/bird-vocalization-classifier/2",
):
    return GoogleBirdVocalizationClassifier(url)


class GoogleBirdVocalizationClassifier(BaseClassifier):
    """load TF model hub google Perch model, wrap in OpSo class

    Args:
        url to model path (default is Google-Bird-Vocalization-Classifier v3)

    Returns:
        object with .predict(), .embed() etc methods

    Methods:
        predict: get per-audio-clip per-class scores in dataframe format; includes WandB logging
        generate_embeddings: make embeddings for audio data (feature vectors from penultimate layer)
        generate_embeddings_and_logits: returns (embeddings, logits)
    """

    def __init__(self, url="https://tfhub.dev/google/bird-vocalization-classifier/2"):
        """load TF model hub google Perch model, wrap in OpSo TensorFlowHubModel class

        Args:
            url to model path (default is Perch v3)

        Returns:
            object with .predict(), .embed() etc methods
        """
        # only require tensorflow and tensorflow_hub if/when this class is used
        try:
            import tensorflow
            import tensorflow_hub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GoogleBirdVocalizationClassifier requires tensorflow and "
                "tensorflow_hub packages to be installed. "
                "Install in your python environment with `pip install tensorflow tensorflow_hub`"
            ) from exc

        self.network = tensorflow_hub.load(url)
        self.preprocessor = AudioPreprocessor(sample_duration=5, sample_rate=32000)
        self.inference_dataloader_cls = AudioSampleArrayDataloader

        # load class list
        resources = Path(__file__).parent.parent / "resources"
        label_csv = resources / "google-bird-vocalization-classifier_v3_classes.csv"
        self.classes = pd.read_csv(label_csv)["ebird2021"].values

    def __call__(
        self, dataloader, return_embeddings=False, return_logits=True, **kwargs
    ):
        """kwargs are passed to SafeAudioDataloader init (num_workers, batch_size, etc)"""

        if not return_logits and not return_embeddings:
            raise ValueError("Both return_logits and return_embeddings cannot be False")

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        for batch in tqdm(dataloader):
            batch_logits, batch_embeddings = self.network.infer_tf(batch)
            logits.extend(batch_logits.numpy().tolist())
            embeddings.extend(batch_embeddings.numpy().tolist())

        if return_logits and return_embeddings:
            return embeddings, logits
        elif return_logits:
            return logits
        elif return_embeddings:
            return embeddings

    def generate_embeddings(self, samples, **kwargs):
        """Generate embeddings for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        Returns:
            list of embeddings
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=True, return_logits=False)

    def generate_logits(self, samples, **kwargs):
        """Return (logits, embeddings) for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=False, return_logits=True)

    def generate_embeddings_and_logits(self, samples, **kwargs):
        """Return (logits, embeddings) for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=True, return_logits=True)
