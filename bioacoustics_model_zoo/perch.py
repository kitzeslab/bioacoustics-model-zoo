from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow
import torch
import warnings
import platform

import opensoundscape
from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape import Action, Audio, CNN

from bioacoustics_model_zoo.utils import (
    collate_to_np_array,
    AudioSampleArrayDataloader,
    register_bmz_model,
)
from bioacoustics_model_zoo.tensorflow_wrapper import (
    TensorFlowModelWithPytorchClassifier,
)


@register_bmz_model
class Perch(TensorFlowModelWithPytorchClassifier):
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

    Note: because TensorFlow Hub implements its own caching system, we do not use the bioacoustics
    model zoo caching functionality here. TF Hub caches to a temporary directory by default (does not
    persist across system restart), but this can be configured
    (see https://www.tensorflow.org/hub/caching#caching_of_compressed_downloads)

    Args:
        version: [default: 8], supports versions 3 & 4 (outputs (logits, embeddings))
            and version 8 (outputs dictionary with keys 'order', 'family',
            'genus', 'label', 'embedding', 'frontend')
        path: (optional, typically leave as None)
            optional url to model path on TensorFlow Hub (default is Perch v8),
            Default: None locates the model on tfhub based on version argument

            OR path to local _folder_ containing /savedmodel/saved_model.pb
            and /label.csv

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
    import bioacoustics_model_zoo as bm
    model=bmz.Perch()
    predictions = model.predict(['test.wav']) #predict on the model's classes
    embeddings = model.embed(['test.wav']) #generate embeddings on each 5 sec of audio
    ```

    Example 2: loading from local folder
    ```
    m = bmz.Perch(path='/path/to/perch_folder/',)
    """

    def __init__(self, version=8, path=None):

        # only require tensorflow and kagglehub if/when this class is used
        try:
            import tensorflow as tf
            import kagglehub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GoogleBirdVocalizationClassifier requires tensorflow and "
                "kagglehub packages to be installed. "
                "Install in your python environment with `pip install tensorflow kagglehub`"
            ) from exc

        self.version = version

        kagglehub_handles = {
            4: "google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/4",
            8: "google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8",
        }

        if path is None:
            path = kagglehub_handles[version]

        # Load pre-trained model: handle url from tfhub or local dir
        if path.startswith("google/"):
            # load from kagglehub
            model_path = kagglehub.model_download(path)
            tf_model = tensorflow.saved_model.load(model_path)

            # load lists of taxonomic levels
            label_csv = Path(model_path) / "assets/label.csv"
            order_csv = Path(model_path) / "assets/order.csv"
            family_csv = Path(model_path) / "assets/family.csv"
            genus_csv = Path(model_path) / "assets/genus.csv"

        else:
            # must be a local directory containing /savedmodel/saved_model.pb and
            # csvs: label.csv, order.csv, family.csv, genus.csv listing class names
            assert Path(path).is_dir(), f"url {path} is not a directory or a URL"
            # tf.saved_model.load looks for `saved_model.pb` file in the directory passed to it
            tf_model = tf.saved_model.load(Path(path) / "savedmodel")
            # load lists of taxonomic levels
            label_csv = Path(path) / "label.csv"
            order_csv = Path(path) / "assets/order.csv"
            family_csv = Path(path) / "assets/family.csv"
            genus_csv = Path(path) / "assets/genus.csv"

        taxonomic_classes = {
            "order": pd.read_csv(order_csv)["ebird2021_orders"].values,
            "family": pd.read_csv(family_csv)["ebird2021_families"].values,
            "genus": pd.read_csv(genus_csv)["ebird2021_genera"].values,
            "species": pd.read_csv(label_csv)["ebird2021"].values,
        }

        # initialize parent class with methods for training custom classifier head
        super().__init__(
            embedding_size=1280,
            classes=taxonomic_classes["species"],
            sample_duration=5,
            sample_rate=32000,
        )
        self.version = version
        self.taxonomic_classes = taxonomic_classes
        self.tf_model = tf_model
        self.inference_dataloader_cls = AudioSampleArrayDataloader
        self.train_dataloader_cls = AudioSampleArrayDataloader
        self._class_outputs_key = "label"

        # match the resampling method used by Perch / HopLite repo
        self.preprocessor.pipeline["load_audio"].params["resample_type"] = "polyphase"

        # perch preprocessing normalizes audio to peak = 0.25
        # https://github.com/kitzeslab/bioacoustics-model-zoo/issues/30#issuecomment-3134186126
        self.preprocessor.insert_action(
            action_index="normalize_signal",
            action=Action(
                opensoundscape.Audio.normalize, is_augmentation=False, peak_level=0.25
            ),
        )

        # if on a mac, disable XLA JIT to avoid TF hanging behavior (as of TF 2.21.0, March 2026)
        if platform.system() == "Darwin":
            warnings.warn(
                "Disabling TensorFlow's XLA compilation (setting tf.config.optimizer.set_jit(False)) because otherwise "
                "TF models on Mac hang at runtime as of Tensorflow 2.21.0"
            )
            tf.config.optimizer.set_jit(False)

    def batch_forward(self, batch_data):
        """run inference on a single batch of samples

        Returned logits depend on self.use_custom_classifier: if False, returns
        logits from the TensorFlow model. If True, returns logits from the
        custom classifier (self.network)

        Args:
            batch_samples: list of AudioSample objects

        Returns:
            dictionary with keys ('order', 'embedding', 'family', 'frontend',
            'genus', 'label', )

            if use_custom_classifier=True:
                dict has additional key 'custom_classifier'
                with outputs of self.network
        """
        batch_data = np.array([s.data.samples for s in batch_data], dtype=np.float32)
        outs = self.tf_model.infer_tf(batch_data)
        if self.version < 8:
            # earlier versions just output (logits, embeddings)
            outs = {
                "label": outs[0],
                "embedding": outs[1],
                "order": None,
                "family": None,
                "genus": None,
                "frontend": None,
            }

        if self.use_custom_classifier:
            # use custom Pytorch classifier head to generate logits
            emb_tensor = torch.tensor(outs["embedding"]).to(self.device)
            self.network.to(self.device)
            # add additional key to output dictionary
            outs["custom_classifier"] = self.network(emb_tensor).detach().cpu().numpy()
        return outs

    def __call__(
        self, dataloader, wandb_session=None, progress_bar=True, targets=None,
    ):
        """run forward pass of model, iterating through dataloader batches, returning dict of aggregated outputs

        Args:
            dataloader: instance of SafeAudioDataloader or custom subclass
            wandb_session: wandb.Session object, if provided, logs progress
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            targets: tuple of strings, specifying which outputs to return. 
                Default is 'label' if self.use_custom_classifier=False, or 'custom_classifier' if self.use_custom_classifier=True
                Options:
                - 'label': logits for species classes (default)
                - 'embedding': feature vector from penultimate layer
                - 'order': logits for order classes (only for version >= 8)
                - 'family': logits for family classes (only for version >= 8)
                - 'genus': logits for genus classes (only for version >= 8)
                - 'frontend': log mel spectrograms generated at the input layer (only for version >= 8)
                - 'custom_classifier': logits from the custom classifier head
                Note:  'order', 'family', 'genus', and 'frontend' are only available from verion >= 8

        Returns: dictionary with keys specified in `targets`, each containing a numpy array of outputs for all samples in the dataloader

        """
        import tensorflow as tf
        if targets is None:
            targets = ("custom_classifier",) if self.use_custom_classifier else ("label",)
        # iterate batches, running inference on each
        results = {key: [] for key in targets}
        for i, batch_samples in enumerate(tqdm(dataloader, disable=not progress_bar)):
            # batch_forward returns dict with tf model outputs + custom classifier
            # logits if self.use_custom_classifier=True
            outs = self.batch_forward(batch_samples)

            # only aggregate the outputs requested to save memory
            for key in targets:
                results[key].append(outs[key])
            
            if wandb_session is not None:
                wandb_session.log(
                    {
                        "progress": i / len(dataloader),
                        "completed_batches": i,
                        "total_batches": len(dataloader),
                    }
                )
        # concatenate results for each key into a single array
        results = { key: tf.concat(value, axis=0).numpy() for key, value in results.items()}
        return results

    def forward(self, samples, progress_bar=True, wandb_session=None, targets = ('label', 'embedding', 'order','family', 'genus', 'frontend'), **dataloader_kwargs):
        """
        Run inference on a list of samples, returning all outputs as a dictionary

        Args:
            samples: list of file paths, OR pd.DataFrame with index containing audio file paths
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            wandb_session: wandb.Session object, if provided, logs progress
            targets: tuple of strings, specifying which outputs to return
                any of: 'label', 'embedding', 'order', 'family', 'genus', 'frontend', 'custom_classifier'
            **dataloader_kwargs: additional keyword arguments passed to self.predict_dataloader()

        Returns: dictionary with keys specified in `targets`, each containing a numpy array of outputs for all samples in the dataloader

        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples, **dataloader_kwargs)

        # run inference, getting all outputs
        outs = self(
            dataloader=dataloader,
            wandb_session=wandb_session,
            progress_bar=progress_bar,
            targets=targets
        )

        # put logit & embedding outputs in DataFrames with multi-index like .predict()
        for k in ["order", "family", "genus", "label", "embedding", "custom_classifier"]:
            if k in outs:
                if outs[k] is not None:
                    outs[k] = pd.DataFrame(
                        data=outs[k],
                        index=dataloader.dataset.dataset.label_df.index,
                        columns=self.taxonomic_classes.get(k, None),
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
        outs = self(
            dataloader=dataloader,
            progress_bar=progress_bar,
            targets=("embedding", "label") if return_preds else ("embedding",)
        )
        embeddings = outs["embedding"]
       
        if return_dfs:
            # put embeddings in DataFrame with multi-index like .predict()
            embeddings = pd.DataFrame(
                data=embeddings, index=dataloader.dataset.dataset.label_df.index
            )

        if return_preds:
            preds = outs["label"]
            if return_dfs:
                # put predictions in a DataFrame with same index as embeddings
                preds = pd.DataFrame(
                    data=preds, index=dataloader.dataset.dataset.label_df.index
                )
            return embeddings, preds
        else:
            return embeddings
