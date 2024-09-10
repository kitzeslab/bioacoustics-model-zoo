from pathlib import Path

import pandas as pd
import numpy as np
import urllib

import opensoundscape
from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape import Action, Audio, CNN

from bioacoustics_model_zoo.utils import (
    collate_to_np_array,
)


class Perch(CNN):
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
        embed: generate embedding layer outputs for samples; optionally return logits as well

    Example 1: download from TFHub and generate logits and embeddings
    ```
    import torch
    model=torch.hub.load('kitzeslab/bioacoustics_model_zoo', 'Perch',trust_repo=True)
    predictions = model.predict(['test.wav']) #predict on the model's classes
    embeddings = model.embed(['test.wav']) #generate embeddings on each 5 sec of audio
    ```

    Example 2: loading from local folder
    ```
    m = torch.hub.load(
        'kitzeslab/bioacoustics-model-zoo:google_bird_model',
        'Perch',
        url='/path/to/perch_0.1.2/',
        trust_repo=True
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
        self.sample_duration = 5

        # extend short samples to 5s by padding end with zeros (silence)
        self.preprocessor.insert_action(
            action_index="extend",
            action=Action(
                Audio.extend_to, is_augmentation=False, duration=self.sample_duration
            ),
        )
        self.inference_dataloader_cls = SafeAudioDataloader

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

        self.device = opensoundscape.ml.cnn._gpu_if_available()

    def __call__(
        self, dataloader, wandb_session=None, progress_bar=True, return_embeddings=False
    ):
        """run forward pass of model, iterating through dataloader batches

        Args:
            dataloader: instance of SafeAudioDataloader or custom subclass

        Returns: logits, or (logits, embeddings) if return_embeddings=True
            logits: numpy array of shape (n_samples, n_classes)
            embeddings: numpy array of shape (n_samples, n_features)
        """
        # iterate batches, running inference on each
        logits = []
        embeddings = []
        # logmelspec = []
        for i, samples_batch in enumerate(tqdm(dataloader, disable=not progress_bar)):
            batch_logits, batch_embeddings = self.network.infer_tf(samples_batch)
            logits.extend(batch_logits.numpy().tolist())
            embeddings.extend(batch_embeddings.numpy().tolist())

            if wandb_session is not None:
                wandb_session.log(
                    {
                        "progress": i / len(dataloader),
                        "completed_batches": i,
                        "total_batches": len(dataloader),
                    }
                )
        if return_embeddings:
            return (np.array(logits), np.array(embeddings))
        else:
            return np.array(logits)

    def predict_dataloader(self, samples, **kwargs):
        """generate dataloader for inference

        kwargs are passed to self.inference_dataloader_class.__init__

        behaves the same as the parent class, except for a custom collate_fn
        """
        return super().predict_dataloader(
            samples=samples, collate_fn=collate_to_np_array, **kwargs
        )

    def embed(
        self,
        samples,
        progress_bar=True,
        return_preds=False,
        return_dfs=True,
        **kwargs,
    ):
        """
        Generate embeddings for audio files/clips

        Args:
            samples: same as CNN.predict(): list of file paths, OR pd.DataFrame with index
                containing audio file paths, OR a pd.DataFrame with multi-index (file, start_time,
                end_time)
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            return_preds: bool, if True, returns two outputs (embeddings, logits)
            return_dfs: bool, if True, returns embeddings as pd.DataFrame with multi-index like
                .predict(). if False, returns np.array of embeddings [default: True].
            kwargs are passed to self.predict_dataloader()

        Returns: (embeddings, preds) if return_preds=True or embeddings if return_preds=False
            types are pd.DataFrame if return_dfs=True, or np.array if return_dfs=False
            - preds are always logit scores with no activation layer applied
            - embeddings are the feature vectors from the penultimate layer of the network
        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples, **kwargs)

        # run inference, returns (scores, intermediate_outputs)
        preds, embeddings = self(
            dataloader=dataloader,
            progress_bar=progress_bar,
            return_embeddings=True,
        )

        if return_dfs:
            # put embeddings in DataFrame with multi-index like .predict()
            embeddings = pd.DataFrame(
                data=embeddings, index=dataloader.dataset.dataset.label_df.index
            )

        if return_preds:
            if return_dfs:
                # put predictions in a DataFrame with same index as embeddings
                preds = pd.DataFrame(
                    data=preds, index=dataloader.dataset.dataset.label_df.index
                )
            return embeddings, preds
        else:
            return embeddings
