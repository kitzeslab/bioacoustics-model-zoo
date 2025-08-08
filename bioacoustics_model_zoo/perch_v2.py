from bioacoustics_model_zoo.utils import register_bmz_model
from bioacoustics_model_zoo.tensorflow_wrapper import (
    TensorFlowModelWithPytorchClassifier,
)

from pathlib import Path

import pandas as pd
import numpy as np
import urllib
import torch
import warnings

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
class Perch2(TensorFlowModelWithPytorchClassifier):
    """load Perch v2.0 from TensorFlow Hub

    [Perch v2](https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2)
    is shared under the [Apache 2.0 License](https://opensource.org/license/apache-2-0/).

    The model can be used to classify sounds from about 15,000 species (10,000 are birds), or
    to generate feature embeddings for audio files. It was trained on recordings from Xeno Canto
    and iNaturalist Sounds.

    The model is not "fine-tunable" - that is, you can train classification heads on the embeddings
    but cannot train the feature extractor weights.

    Model performance is described in :
    ```
    Bart van Merriënboer, Vincent Dumoulin, Jenny Hamer, Lauren Harrell, Andrea
    Burns and Tom Denton, preprint 2025. "Perch 2.0: The Bittern Lesson for
    Bioacoustics." biorxiv: https://arxiv.org/pdf/2508.04665
    ```

    Note: because TensorFlow Hub implements its own caching system, we do not use the bioacoustics
    model zoo caching functionality here. TF Hub caches to a temporary directory by default (does not
    persist across system restart), but this can be configured
    (see https://www.tensorflow.org/hub/caching#caching_of_compressed_downloads)

    Args:
        version: [default: 1] currently only supported version is 1 (Perch v2.0)

    Methods:
        predict: get per-audio-clip per-class scores as pandas DataFrame
        embed: generate embedding layer outputs for samples
        forward: return all outputs as a dictionary with keys:


    Example:
    ```
    import bioacoustics_model_zoo as bm
    model=bmz.Perch2()
    predictions = model.predict(['test.wav']) #predict on the model's classes
    embeddings = model.embed(['test.wav']) #generate embeddings on each 5 sec of audio
    all_outputs = model.forward(['test.wav'], return_value='all') #get all model outputs
    all_outputs['spatial_embedding'].shape # np.array of spatial embeddings
    ```

    Environment setup: currently only working on Linux with TensorFlow 2.20.0rc0
    ```
    pip install --upgrade opensoundscape
    pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo@perch2
    pip install tensorflow==2.20.0rc0 tensorflow-hub
    pip install --no-deps tf-keras==0.19.0
    ```

    """

    def __init__(self, version=1):
        # only require tensorflow and tensorflow_hub if/when this class is used
        try:
            import tensorflow_hub as hub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                """Perch2 requires tensorflow and tensorflow_hub packages to be installed. 
                Install in your python environment with 
                `pip install tensorflow[and-cuda]~=2.20.0rc0` (for linux w GPU support)
                or `pip install tensorflow~=2.20.0rc0`
                """
            ) from exc

        tested_versions = (1,)
        if not version in tested_versions:
            warnings.warn(
                f"version {version} has not been tested, tested versions: {tested_versions}"
            )
        self.version = version

        tfhub_path = f"https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/{version}"

        from pathlib import Path
        import pandas as pd

        # first try hub.load(url): succeeds to download but fails to find file within subfolder
        try:
            tf_model = hub.load(tfhub_path)
        except Exception as e:
            # download may have succeeded
            # try to open from downloaded local path
            local_path = Path(hub.resolve(tfhub_path)) / "perch_v2"
            tf_model = hub.load(str(local_path))  # or tf.saved_model.load()
        model_path = hub.resolve(tfhub_path)
        class_lists_glob = (Path(model_path) / "perch_v2/assets").glob("*.csv")
        class_lists = {}
        for class_list_path in class_lists_glob:
            try:
                class_list_name = class_list_path.stem
                df = pd.read_csv(class_list_path)
                namespace = df.columns[0]
                classes = df[namespace].tolist()
                class_lists[class_list_name] = {
                    "namespace": namespace,
                    "classes": classes,
                }
            except:
                print(
                    f"failed to read class list from {class_list_path.stem}, skipping"
                )

        # initialize parent class with methods for training custom classifier head
        super().__init__(
            embedding_size=1536,
            classes=class_lists["labels"]["classes"],
            sample_duration=5,
        )
        self.version = version
        self.ebird_codes = class_lists["perch_v2_ebird_classes"]["classes"]
        self.tf_model = tf_model
        self.inference_dataloader_cls = AudioSampleArrayDataloader
        self.train_dataloader_cls = AudioSampleArrayDataloader

        # Configure preprocessing
        # Perch expects audio signal input as 32kHz mono 5s clips (160,000 samples)
        self.preprocessor = AudioAugmentationPreprocessor(
            sample_duration=5, sample_rate=32000
        )
        self.sample_duration = 5

        # extend short samples to 5s by padding end with zeros (silence)
        self.preprocessor.insert_action(
            action_index="extend",
            action=Action(
                Audio.extend_to, is_augmentation=False, duration=self.sample_duration
            ),
        )

        # match the resampling method used by Perch / HopLite repo
        self.preprocessor.pipeline["load_audio"].params["resample_type"] = "polyphase"

        # avoid invalid sample values outside of [-1,1]
        self.preprocessor.insert_action(
            action_index="normalize_signal",
            action=Action(
                opensoundscape.Audio.normalize, is_augmentation=False, peak_level=0.25
            ),
        )

    def _batch_forward(self, batch_data, return_dict=False):
        """run inference on a single batch of samples

        Returned logits depend on self.use_custom_classifier: if False, returns
        logits from the TensorFlow model. If True, returns logits from the
        custom classifier (self.network)

        Args:
            batch_data: np.array of audio samples, shape (batch_size, 32000*5)
            return_dict: bool, if True, returns a dictionary of all outputs
                [default: False] return (embeddings, logits)
                if True, returns dictionary with keys:
                ['embedding', 'spatial_embedding', 'label', 'spectrogram']

        Returns:
            if return_dict=False: tuple (embeddings, logits)
            if return_dict=True: dictionary with keys:
                ['embedding', 'spatial_embedding', 'label', 'spectrogram']
        """

        model_outputs = self.tf_model.signatures["serving_default"](inputs=batch_data)

        # move tensorflow tensors to CPU and convert to numpy
        outs = {k: None if v is None else v.numpy() for k, v in model_outputs.items()}

        if self.use_custom_classifier:
            # use custom classifier to generate logits
            emb_tensor = torch.tensor(outs["embedding"]).to(self.device)
            self.network.to(self.device)
            logits = self.network(emb_tensor).detach().cpu().numpy()
            # add additional key to output dictionary
            outs["custom_classifier"] = logits
        else:
            logits = outs["label"]

        if return_dict:
            return outs
        else:
            return outs["embedding"], logits

    def __call__(
        self, dataloader, wandb_session=None, progress_bar=True, return_value="logits"
    ):
        """run forward pass of model, iterating through dataloader batches

        The output can take several forms since it depends both on `return_value` and
        on self.use_custom_classifier. See details below.

        Args:
            dataloader: instance of SafeAudioDataloader or custom subclass
            wandb_session: wandb.Session object, if provided, logs progress
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            return_value: str, 'logits', 'embeddings', 'spatial_embeddings', or 'all'
                [default: 'logits']
                - 'logits': returns only the logits on the species classes
                - 'embeddings': returns only the feature vectors from the penultimate layer
                    embedding shape: (1536,)
                - 'spatial_embeddings': returns only the spatial embeddings from the model
                    (embedding shape: (5, 3, 1536))
                - 'all': returns dictionary with all Perch outputs:
                    keys: ['embedding', 'spatial_embedding', 'label', 'spectrogram']
                Note: if self.custom_classifier=True:
                - for return_value='logits', returns logits from self.network
                - for return_value='all', returns dictionary with additional key
                'custom_classifier' with logits from self.network

        Returns: depends on return_value argument (see above)

        """
        # check return_value argument is valid
        return_value_options = ("logits", "embeddings", "spatial_embeddings", "all")
        err = f"return_value must be one of {return_value_options}, got {return_value}"
        assert return_value in return_value_options

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        spatial_embeddings = []
        spectrograms = []
        custom_classifier_logits = []
        for i, (samples_batch, _) in enumerate(
            tqdm(dataloader, disable=not progress_bar)
        ):
            # _batch_forward returns dict with tf model outputs + custom classifier
            # logits if self.use_custom_classifier=True
            outs = self._batch_forward(samples_batch, return_dict=True)

            # only aggregate the outputs requested to save memory
            if return_value in ("all", "logits"):
                logits.extend(outs["label"].tolist())
                if self.use_custom_classifier:
                    custom_classifier_logits.extend(outs["custom_classifier"].tolist())
            if return_value in ("all", "embeddings"):
                embeddings.extend(outs["embedding"].tolist())
            if return_value in ("all", "spatial_embeddings"):
                spatial_embeddings.extend(outs["spatial_embedding"].tolist())
            if return_value == "all":
                spectrograms.extend(outs["spectrogram"].tolist())

            if wandb_session is not None:
                wandb_session.log(
                    {
                        "progress": i / len(dataloader),
                        "completed_batches": i,
                        "total_batches": len(dataloader),
                    }
                )
        if return_value == "all":
            return {
                "label": np.array(logits),
                "embedding": np.array(embeddings),
                "spatial_embedding": np.array(spatial_embeddings),
                "spectrogram": np.array(spectrograms),
                "custom_classifier": (
                    np.array(custom_classifier_logits)
                    if self.use_custom_classifier
                    else None
                ),
            }
        elif return_value == "embeddings":
            return np.array(embeddings)
        elif return_value == "logits":
            return (
                np.array(custom_classifier_logits)
                if self.use_custom_classifier
                else np.array(logits)
            )
        elif return_value == "spatial_embeddings":
            return np.array(spatial_embeddings)

    def forward(
        self, samples, progress_bar=True, wandb_session=None, return_value="all"
    ):
        """
        Run inference on a list of samples, returning all outputs as a dictionary

        wraps self.predict_dataloader() and self.__call__() to run the model on audio files/clips

        Args:
            samples: list of file paths, OR pd.DataFrame with index containing audio file paths
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            wandb_session: wandb.Session object, if provided, logs progress
            return_value: str, one of 'logits', 'embeddings', 'spatial_embeddings', or 'all'
                [default: 'all'] returns all outputs as a dictionary, otherwise
                returns only the requested output type

        Returns: dictionary with keys ('order', 'embedding', 'family', 'frontend', 'genus', 'label'):
            - order, family, genus, and label (species) are dataframes of logits at different taxonomic levels
            - embedding is a dataframe with the feature vector from the penultimate layer
            - frontend is np.array of log mel spectrograms generated at the input layer
            - if self.use_custom_classifier, includes additional key 'custom_classifier'
                with a dataframe containing the outputs of self.network(embeddings)
        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples)

        # run inference, getting all outputs
        return self(
            dataloader=dataloader,
            wandb_session=wandb_session,
            progress_bar=progress_bar,
            return_value=return_value,
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
