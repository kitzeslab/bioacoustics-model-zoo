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
        self.inference_dataloader_cls = (
            SafeAudioDataloader  # AudioSampleArrayDataloader
        )
        self.sample_duration = 5

        # load class list
        resources = Path(__file__).parent.parent / "resources"
        label_csv = resources / "google-bird-vocalization-classifier_v3_classes.csv"
        self.classes = pd.read_csv(label_csv)["ebird2021"].values

    def __call__(self, dataloader, **kwargs):
        """kwargs are passed to SafeAudioDataloader init (num_workers, batch_size, etc)

        returns logits, embeddings, start_times, files"""

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        logmelspec = []
        start_times = []
        files = []
        for batch in tqdm(dataloader):
            samples_batch = collate_to_np_array(batch)
            batch_logits, batch_embeddings = self.network.infer_tf(samples_batch)
            logits.extend(batch_logits.numpy().tolist())
            embeddings.extend(batch_embeddings.numpy().tolist())
            start_times = [s.start_time for s in batch]
            files = [s.source for s in batch]
        return (np.array(i) for i in (logits, embeddings, start_times, files))

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
        _, embeddings, start_times, files = self(dataloader)
        end_times = start_times + self.sample_duration
        return pd.DataFrame(
            embeddings,
            index=pd.MultiIndex.from_arrays(
                [files, start_times, end_times],
                names=["file", "start_time", "end_time"],
            ),
        )

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
        logits, _, start_times, files = self(dataloader)
        end_times = start_times + self.sample_duration
        return pd.DataFrame(
            logits,
            index=pd.MultiIndex.from_arrays(
                [files, start_times, end_times],
                names=["file", "start_time", "end_time"],
            ),
            columns=self.classes,
        )

    def predict(self, samples, **kwargs):
        """alias for generate_logits()"""
        return self.generate_logits(samples, **kwargs)

    def generate_embeddings_and_logits(self, samples, **kwargs):
        """Return (logits, embeddings) for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        returns 2 dataframes: (logits, embeddings)
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)

        logits, embeddings, start_times, files = self(dataloader)
        end_times = start_times + self.sample_duration
        logits_df = pd.DataFrame(
            logits,
            index=pd.MultiIndex.from_arrays(
                [files, start_times, end_times],
                names=["file", "start_time", "end_time"],
            ),
            columns=self.classes,
        )
        embeddings_df = pd.DataFrame(
            embeddings,
            index=pd.MultiIndex.from_arrays(
                [files, start_times, end_times],
                names=["file", "start_time", "end_time"],
            ),
        )
        return logits_df, embeddings_df
