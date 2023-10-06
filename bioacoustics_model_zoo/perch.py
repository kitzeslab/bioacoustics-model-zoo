from pathlib import Path

import pandas as pd
import numpy as np
import urllib

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier


from bioacoustics_model_zoo.utils import (
    collate_to_np_array,
)


class Perch(BaseClassifier):
    """load Perch (aka Google Bird Vocalization Classifier) from TensorFlow Hub or local file

    [Perch](https://tfhub.dev/google/bird-vocalization-classifier/4) is shared under the
    [Apache 2.0 License](https://opensource.org/license/apache-2-0/).

    The model can be used to classify bird vocalizations from about 10,000 bird species, or
    to generate feature embeddings for audio files. It was trained on recordings from Xeno Canto.

    Model performance is described in :
    ```
    Ghani, Burooj, et al.
    "Feature embeddings from large-scale acoustic bird classifiers enable few-shot transfer learning."
    arXiv preprint arXiv:2307.06292 (2023).
    ```

    Args:
        model_dir: path to local folder containing /savedmodel/saved_model.pb and /label.csv

    Returns:
        object with .predict(), .generate_embeddings() etc methods

    Methods:
        predict: get per-audio-clip per-class scores as pandas DataFrame
        generate_logits: equivalent to predict()
        generate_embeddings: make df of embeddings (feature vectors from penultimate layer)
        generate_embeddings_and_logits: returns 2 dfs: (embeddings, logits)

    Example 1: download from TFHub and generate logits and embeddings
    ```
    import torch
    model=torch.hub.load('kitzeslab/bioacoustics_model_zoo', 'Perch')
    predictions = model.predict(['test.wav']) #predict on the model's classes
    embeddings = model.generate_embeddings(['test.wav']) #generate embeddings on each 5 sec of audio
    ```

    Example 2: loading from local folder
    ```
    m = torch.hub.load(
        'kitzeslab/bioacoustics-model-zoo:google_bird_model',
        'Perch',
        url='/path/to/perch_0.1.2/',
    )
    """

    def __init__(self, url="https://tfhub.dev/google/bird-vocalization-classifier/4"):
        """load TF model hub google Perch model, wrap in OpSo TensorFlowHubModel class

        Args:
            url: url to model path on TensorFlow Hub (default is Perch v4),
                OR path to local folder containing /savedmodel/saved_model.pb and /label.csv

        Returns:
            object with .predict(), .embed() etc methods
        """
        # only require tensorflow and tensorflow_hub if/when this class is used
        # this didn't work - could put try/except block at top, then check tf is availble here
        try:
            import tensorflow as tf
            import tensorflow_hub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GoogleBirdVocalizationClassifier requires tensorflow and "
                "tensorflow_hub packages to be installed. "
                "Install in your python environment with `pip install tensorflow tensorflow_hub`"
            ) from exc

        self.preprocessor = AudioPreprocessor(sample_duration=5, sample_rate=32000)
        self.inference_dataloader_cls = SafeAudioDataloader
        self.sample_duration = 5

        # Load pre-trained model: handle url from tfhub or local dir
        if urllib.parse.urlparse(url).scheme in ("http", "https"):
            # its a url, load from tfhub
            self.network = tensorflow_hub.load(url)

            # load class list
            label_csv = tensorflow_hub.resolve(url) + "/assets/label.csv"
            self.classes = pd.read_csv(label_csv)["ebird2021"].values
        else:
            # must be a local directory containing /savedmodel/saved_model.pb and /label.csv
            assert Path(url).is_dir(), f"url {url} is not a directory or a URL"
            # tf.saved_model.load looks for `saved_model.pb` file in the directory passed to it
            self.network = tf.saved_model.load(Path(url) / "savedmodel")
            # load class list
            self.classes = pd.read_csv(Path(url) / "label.csv")["ebird2021"].values

    def __call__(self, dataloader):
        """run forward pass of model, iterating through dataloader batches

        Args:
            dataloader: instance of SafeAudioDataloader or custom subclass

        Returns:
            (logits, embeddings, start_times, files)

        """

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
            start_times.extend([s.start_time for s in batch])
            files.extend([s.source for s in batch])
        return (np.array(i) for i in (logits, embeddings, start_times, files))

    def generate_embeddings(self, samples, **kwargs):
        """Generate embeddings for audio data

        kwargs are passed to SafeAudioDataloader init (num_workers, batch_size, etc)

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

        kwargs are passed to SafeAudioDataloader init (num_workers, batch_size, etc)

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

    # predict is alias for generate_logits
    predict = generate_logits

    def generate_embeddings_and_logits(self, samples, **kwargs):
        """Return (embeddings, logits) dataframes for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        returns 2 dataframes: (embeddings, logits)
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
        return embeddings_df, logits_df
