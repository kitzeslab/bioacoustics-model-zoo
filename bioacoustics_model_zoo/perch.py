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

    def __init__(self, version=8, path=None):
        """load Perch (aka Google Bird Vocalization Classifier) from TensorFlow Hub or local file

        [Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8)
        is shared under the [Apache 2.0 License](https://opensource.org/license/apache-2-0/).

        The model can be used to classify bird vocalizations from about 10,000 bird species, or
        to generate feature embeddings for audio files. It was trained on recordings from Xeno Canto.

        Model performance is described in :
        ```
        Ghani, Burooj, et al.
        "Feature embeddings from large-scale acoustic bird classifiers enable few-shot transfer learning."
        arXiv preprint arXiv:2307.06292 (2023).
        ```

        Args:
            version: [default: 8], supports versions 3 & 4 (outputs (logits, embeddings))
                and version 8 (outputs dictionary with keys 'order', 'family',
                'genus', 'label', 'embedding', 'frontend')
            path: url to model path on TensorFlow Hub (default is Perch v4),
                OR path to local _folder_ containing /savedmodel/saved_model.pb
                and /label.csv
                [default: None loads from TFHub based on version]
                Note: adjust `version` argument to match the model version in the local folder

        Methods:
            predict: get per-audio-clip per-class scores as pandas DataFrame
            embed: generate embedding layer outputs for samples
            forward: return all outputs as a dictionary with keys
                ('order', 'embedding', 'family', 'frontend', 'genus', 'label'):
                - order, family, genus, and label (species) are logits at
                    different taxonomic levels
                - embedding is the feature vector from the penultimate layer
                - frontend is the log mel spectrogram generated at the input
                    layer

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
            'kitzeslab/bioacoustics-model-zoo',
            'Perch',
            url='/path/to/perch_folder/',
            trust_repo=True
        )
        """
        # only require tensorflow and tensorflow_hub if/when this class is used
        try:
            import tensorflow as tf
            import tensorflow_hub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GoogleBirdVocalizationClassifier requires tensorflow and "
                "tensorflow_hub packages to be installed. "
                "Install in your python environment with `pip install tensorflow tensorflow_hub`"
            ) from exc

        tfhub_paths = {
            4: "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4",
            8: "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8",
        }
        self.version = version

        if path is None:
            path = tfhub_paths[version]

        # Perch expects audio signal input as 32kHz mono 5s clips (160,000 samples)
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
        if urllib.parse.urlparse(path).scheme in ("http", "https"):
            # its a url, load from tfhub
            self.network = tensorflow_hub.load(path)

            # load lists of taxonomic levels
            label_csv = tensorflow_hub.resolve(path) + "/assets/label.csv"
            order_csv = tensorflow_hub.resolve(path) + "/assets/order.csv"
            family_csv = tensorflow_hub.resolve(path) + "/assets/family.csv"
            genus_csv = tensorflow_hub.resolve(path) + "/assets/genus.csv"

        else:
            # must be a local directory containing /savedmodel/saved_model.pb and
            # csvs: label.csv, order.csv, family.csv, genus.csv listing class names
            assert Path(path).is_dir(), f"url {path} is not a directory or a URL"
            # tf.saved_model.load looks for `saved_model.pb` file in the directory passed to it
            self.network = tf.saved_model.load(Path(path) / "savedmodel")
            # load lists of taxonomic levels
            label_csv = Path(path) / "label.csv"
            order_csv = Path(path) / "assets/order.csv"
            family_csv = Path(path) / "assets/family.csv"
            genus_csv = Path(path) / "assets/genus.csv"

        self.taxonomic_classes = {
            "order": pd.read_csv(order_csv)["ebird2021_orders"].values,
            "family": pd.read_csv(family_csv)["ebird2021_families"].values,
            "genus": pd.read_csv(genus_csv)["ebird2021_genera"].values,
            "species": pd.read_csv(label_csv)["ebird2021"].values,
        }
        self.classes = self.taxonomic_classes["species"]

        self.device = opensoundscape.ml.cnn._gpu_if_available()

    def __call__(
        self, dataloader, wandb_session=None, progress_bar=True, return_value="logits"
    ):
        """run forward pass of model, iterating through dataloader batches

        Args:
            dataloader: instance of SafeAudioDataloader or custom subclass
            wandb_session: wandb.Session object, if provided, logs progress
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            return_value: str, 'logits', 'embeddings', 'taxonomic', or 'all' [default: 'logits']
                - 'logits': returns only the logits on the species classes
                - 'embeddings': returns only the feature vectors from the penultimate layer
                - 'taxonomic': returns dictionionary with logits on all taxonomic levels
                    (keys: 'order', 'family', 'genus', 'label')
                - 'all': returns dictionary with all Perch outputs
                    (keys: 'order', 'family', 'genus', 'label', 'embedding', 'frontend')
                    - 'label' is the logits on the species classes
                    - 'embedding' is the feature vectors from the penultimate layer
                    - 'frontend' is the log mel spectrogram generated at the input layer
                Note: for version < 8, 'order', 'family', 'genus', and 'frontend' will be None


        Returns: logits, or (logits, embeddings) if return_embeddings=True
            logits: numpy array of shape (n_samples, n_classes)
            embeddings: numpy array of shape (n_samples, n_features)
        """
        # check return_value argument is valid
        err = f"return_value must be one of 'logits', 'embeddings', 'taxonomic', or 'all', got {return_value}"
        assert return_value in ("logits", "embeddings", "taxonomic", "all"), err

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        order_logits = []
        family_logits = []
        genus_logits = []
        logmelspec = []
        for i, samples_batch in enumerate(tqdm(dataloader, disable=not progress_bar)):
            outs = self.network.infer_tf(samples_batch)
            if self.version < 8:
                # earlier versions just output (logits, embeddings)
                if return_value != "embeddings":
                    logits.extend(outs[0].numpy().tolist())
                if return_value != "logits":
                    embeddings.extend(outs[1].numpy().tolist())
            else:
                # only aggregate the outputs requested to save memory
                if return_value != "embeddings":
                    logits.extend(outs["label"].numpy().tolist())
                if return_value != "logits":
                    embeddings.extend(outs["embedding"].numpy().tolist())
                if return_value in ("taxonomic", "all"):
                    order_logits.extend(outs["order"].numpy().tolist())
                    family_logits.extend(outs["family"].numpy().tolist())
                    genus_logits.extend(outs["genus"].numpy().tolist())
                if return_value == "all":
                    logmelspec.extend(outs["frontend"].numpy().tolist())

            if wandb_session is not None:
                wandb_session.log(
                    {
                        "progress": i / len(dataloader),
                        "completed_batches": i,
                        "total_batches": len(dataloader),
                    }
                )
        if return_value == "taxonomic":
            return {
                "order": np.array(order_logits) if self.version >= 8 else None,
                "family": np.array(family_logits) if self.version >= 8 else None,
                "genus": np.array(genus_logits) if self.version >= 8 else None,
                "label": np.array(logits),
            }
        elif return_value == "all":
            return {
                "order": np.array(order_logits) if self.version >= 8 else None,
                "family": np.array(family_logits) if self.version >= 8 else None,
                "genus": np.array(genus_logits) if self.version >= 8 else None,
                "label": np.array(logits),
                "embedding": np.array(embeddings),
                "frontend": np.array(logmelspec) if self.version >= 8 else None,
            }
        elif return_value == "embeddings":
            return np.array(embeddings)
        elif return_value == "logits":
            return np.array(logits)

    def predict_dataloader(self, samples, **kwargs):
        """generate dataloader for inference

        kwargs are passed to self.inference_dataloader_class.__init__

        behaves the same as the parent class, except for a custom collate_fn
        """
        return super().predict_dataloader(
            samples=samples, collate_fn=collate_to_np_array, **kwargs
        )

    def forward(self, samples, progress_bar=True, wandb_session=None):
        """
        Run inference on a list of samples, returning all outputs as a dictionary

        Args:
            samples: list of file paths, OR pd.DataFrame with index containing audio file paths
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            wandb_session: wandb.Session object, if provided, logs progress

        Returns: dictionary with keys ('order', 'embedding', 'family', 'frontend', 'genus', 'label'):
            - order, family, genus, and label (species) are dataframes of logits at different taxonomic levels
            - embedding is a dataframe with the feature vector from the penultimate layer
            - frontend is np.array of log mel spectrograms generated at the input layer
        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples)

        # run inference, getting all outputs
        outs = self(
            dataloader=dataloader,
            wandb_session=wandb_session,
            progress_bar=progress_bar,
            return_value="all",
        )

        # put logit & embedding outputs in DataFrames with multi-index like .predict()
        outs["order"] = (
            pd.DataFrame(
                data=outs["order"],
                index=dataloader.dataset.dataset.label_df.index,
                columns=self.taxonomic_classes["order"],
            )
            if self.version >= 8
            else None
        )
        outs["family"] = (
            pd.DataFrame(
                data=outs["family"],
                index=dataloader.dataset.dataset.label_df.index,
                columns=self.taxonomic_classes["family"],
            )
            if self.version >= 8
            else None
        )
        outs["genus"] = (
            pd.DataFrame(
                data=outs["genus"],
                index=dataloader.dataset.dataset.label_df.index,
                columns=self.taxonomic_classes["genus"],
            )
            if self.version >= 8
            else None
        )
        outs["label"] = pd.DataFrame(
            data=outs["label"],
            index=dataloader.dataset.dataset.label_df.index,
            columns=self.taxonomic_classes["species"],
        )
        outs["embedding"] = pd.DataFrame(
            data=outs["embedding"],
            index=dataloader.dataset.dataset.label_df.index,
        )

        return outs

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

        See also:
            - predict: get per-audio-clip species outputs as pandas DataFrame
            - forward: return all outputs as a dictionary with keys
        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples, **kwargs)

        # run inference, getting embeddings and optionally logits
        if return_preds:
            # get all outputs, then extract embeddings and logits
            outs = self(
                dataloader=dataloader,
                progress_bar=progress_bar,
                return_value="all",
            )
            embeddings = outs["embedding"]
            preds = outs["label"]
        else:
            # save memory by only aggergating the embeddigs
            embeddings = self(
                dataloader=dataloader,
                progress_bar=progress_bar,
                return_value="embeddings",
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
