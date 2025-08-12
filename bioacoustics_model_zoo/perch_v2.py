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
        version: [default: 2] select from released versions of Perch v2.0 on Kaggle
            Note that this is not the "2" in Perch2, but the version of the Perch2 release.

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
    all_outputs = model.forward(['test.wav']) #get all model outputs including spectrograms and spatial embeddings
    all_outputs['spatial_embedding'].shape # np.array of spatial embeddings
    ```

    Environment setup: currently only working on Linux with TensorFlow 2.20.0rc0
    Once stable versions of Tensorflow >=2.20.0 are available, they should also work
    (likewise for tf-keras >=0.20.0)
    ```
    pip install --upgrade opensoundscape
    pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo@perch2
    pip install tensorflow==2.20.0rc0[and-cuda] tensorflow-hub
    pip install --no-deps tf-keras==0.19.0
    ```

    """

    def __init__(self, version=2):
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

        tested_versions = (2,)
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
            raise RuntimeError(
                f"Failed to load Perch2 model from TensorFlow Hub at {tfhub_path}. "
            ) from e
        model_path = hub.resolve(tfhub_path)
        class_lists_glob = (Path(model_path) / "assets").glob("*.csv")
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

    def _batch_forward(
        self, batch_data, return_dict=False, apply_custom_classifier=False
    ):
        """run inference on a single batch of samples

        Returned logits depend on self.use_custom_classifier: if False, returns
        logits from the TensorFlow model. If True, returns logits from the
        custom classifier (self.network)

        Args:
            batch_data: np.array of audio samples, shape (batch_size, 32000*5)
            return_dict: bool, if True, returns a dictionary of all outputs
                [default: False] return (embeddings, logits)
                if True, returns dictionary with keys:
                ['embedding', 'spatial_embedding', 'label', 'spectrogram','custom_classifier_logits']
            apply_custom_classifier: [default: False] if True, run self.network(embeddings)
                to generate 'custom_classifier_logits' output

        Returns:
            if return_dict=False: tuple (embeddings, logits)
            if return_dict=True: dictionary with keys:
                ['embedding', 'spatial_embedding', 'label', 'spectrogram','custom_classifier_logits']
        """
        model_outputs = self.tf_model.signatures["serving_default"](inputs=batch_data)

        # move tensorflow tensors to CPU and convert to numpy
        outs = {k: None if v is None else v.numpy() for k, v in model_outputs.items()}

        if apply_custom_classifier:
            emb_tensor = torch.tensor(outs["embedding"]).to(self.device)
            self.network.to(self.device)
            outs["custom_classifier_logits"] = (
                self.network(emb_tensor).detach().cpu().numpy()
            )

        if return_dict:
            return outs
        else:  # return embeddings, logits
            # which logits depends on self.use_custom_classifier
            # (True: custom classifier ie self.network, False: original Perch2 logits)
            if self.use_custom_classifier:
                return outs["embedding"], outs["custom_classifier_logits"]
            return outs["embedding"], outs["label"]

    def __call__(
        self,
        dataloader,
        wandb_session=None,
        progress_bar=True,
        return_values=None,
    ):
        """run forward pass of model, iterating through dataloader batches

        The output can take several forms since it depends both on `return_value` and
        on self.use_custom_classifier. See details below.

        Args:
            dataloader: instance of SafeAudioDataloader or custom subclass
            wandb_session: wandb.Session object, if provided, logs progress
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            return_values: tuple(str,): select from 'logits', 'embeddings', 'spatial_embeddings','spectrograms'
                [default:  None] returns logits from custom classifier (self.network) if
                    self.use_custom_classifier=True, otherwise returns Perch2 class logits
                Include any combination of the following:
                - 'logits': logit scores (class predictions) on the species classes
                - 'embeddings': 1D feature vectors from the penultimate layer of the network
                - 'spatial_embeddings': un-pooled spatial embeddings from the network
                - 'spectrograms': log-mel spectrograms generated during preprocessing
                - 'custom_classifier_logits': outputs of the custom classifier head

        Returns:
            - return_values=None: returns outputs of custom classifier
                (if self.use_custom_classifier=True), otherwise returns Perch2 logits
            - return_values specified: dictionary of requested outputs, with keys matching return_values


        """
        # check return_value argument is valid
        return_value_options = (
            "logits",
            "embeddings",
            "spatial_embeddings",
            "spectrograms",
            "custom_classifier_logits",
        )
        if return_values is None:
            if self.use_custom_classifier:
                _return_values = ("custom_classifier_logits",)
            else:
                _return_values = ("logits",)
        else:  # check return values are from supported list
            assert all(
                rv in return_value_options for rv in return_values
            ), f"return_values must be a tuple with any of {return_value_options}, got {return_values}"
            _return_values = return_values

        # iterate batches, running inference on each
        # only aggregate requested outputs to save memory
        returns = {k: [] for k in return_value_options if k in _return_values}
        for i, (samples_batch, _) in enumerate(
            tqdm(dataloader, disable=not progress_bar)
        ):
            # _batch_forward returns dict with tf model outputs + custom classifier
            # logits if self.use_custom_classifier=True
            outs = self._batch_forward(
                samples_batch,
                return_dict=True,
                apply_custom_classifier=("custom_classifier_logits" in _return_values),
            )

            # only aggregate the outputs requested to save memory
            if "logits" in _return_values:
                returns["logits"].extend(outs["label"].tolist())
            if "custom_classifier_logits" in _return_values:
                returns["custom_classifier_logits"].extend(
                    outs["custom_classifier_logits"].tolist()
                )
            if "embeddings" in _return_values:
                returns["embeddings"].extend(outs["embedding"].tolist())
            if "spatial_embeddings" in _return_values:
                returns["spatial_embeddings"].extend(outs["spatial_embedding"].tolist())
            if "spectrograms" in _return_values:
                returns["spectrograms"].extend(outs["spectrogram"].tolist())

            if wandb_session is not None:
                wandb_session.log(
                    {
                        "progress": i / len(dataloader),
                        "completed_batches": i,
                        "total_batches": len(dataloader),
                    }
                )

        if return_values is None:  # just return one set of values
            if self.use_custom_classifier:
                return np.array(returns["custom_classifier_logits"])
            else:
                return np.array(returns["logits"])
        return {k: np.array(v) for k, v in returns.items()}

    def forward(
        self,
        samples,
        progress_bar=True,
        wandb_session=None,
        return_values=("logits", "embeddings", "spatial_embeddings", "spectrograms"),
        return_dfs=True,
        **dataloader_kwargs,
    ):
        """
        Run inference on a list of samples, returning all outputs as a dictionary

        wraps self.predict_dataloader() and self.__call__() to run the model on audio files/clips

        then optionally places 1D outputs in dataframes

        Args:
            samples: list of file paths, OR pd.DataFrame with index containing audio file paths
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            wandb_session: wandb.Session object, if provided, logs progress
            return_values: tuple(str,): select from 'logits', 'embeddings',
                'spatial_embeddings', 'spectrograms', 'custom_classifier_logits'
                [default: ('logits','embeddings','spatial_embeddings','spectrograms')]
                Include any combination of the following:
                - 'logits': logit scores (class predictions) on the species classes
                - 'embeddings': 1D feature vectors from the penultimate layer of the network
                - 'spatial_embeddings': un-pooled spatial embeddings from the network
                - 'spectrograms': log-mel spectrograms generated during preprocessing
                - 'custom_classifier_logits': outputs of the custom classifier head (self.network)
            return_dfs: bool, if True, returns outputs as pd.DataFrame with multi-index like
                .predict() ('file','start_time','end_time'), if False, returns np.array
                [default: True]
            **dataloader_kwargs: additional keyword arguments passed to the dataloader such
                as batch_size, num_workers, etc.

        Returns: dictionary with content depending on return_values and return_dfs arguments:
            - 'logits': pd.DataFrame or np.array of per-clip logits on species classes
                shape: (num_clips, num_classes)
            - 'embeddings': pd.DataFrame or np.array of per-clip 1D feature vectors
                shape: (num_clips, 1536)
            - 'spatial_embeddings': np.array of per-clip spatial embeddings
                shape: (num_clips, 5, 3, 1536)
            - 'custom_classifier_logits': pd.DataFrame or np.array of per-clip logits from the
                custom classifier head, shape: (num_clips, num_custom_classes)
        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples, **dataloader_kwargs)

        # run inference, getting all outputs
        # avoids aggregating unrequested outputs to save memory
        results_dict = self(
            dataloader=dataloader,
            wandb_session=wandb_session,
            progress_bar=progress_bar,
            return_values=return_values,
        )

        # optionally put 1D outputs in DataFrames with multi-index ('file','start_time','end_time')
        if return_dfs:
            if "logits" in results_dict:
                results_dict["logits"] = pd.DataFrame(
                    data=results_dict["logits"],
                    index=dataloader.dataset.dataset.label_df.index,
                    columns=self._original_classes,
                )
            if "embeddings" in results_dict:
                results_dict["embeddings"] = pd.DataFrame(
                    data=results_dict["embeddings"],
                    index=dataloader.dataset.dataset.label_df.index,
                    columns=None,
                )
            if "custom_classifier_logits" in results_dict:
                results_dict["custom_classifier_logits"] = pd.DataFrame(
                    data=results_dict["custom_classifier_logits"],
                    index=dataloader.dataset.dataset.label_df.index,
                    columns=self._custom_classes,
                )

        return results_dict

    def embed(
        self,
        samples,
        progress_bar=True,
        return_preds=False,
        return_dfs=True,
        wandb_session=None,
        **dataloader_kwargs,
    ):
        """
        Generate embeddings for audio files/clips

        Matches API of opensoundscape.SpectrogramClassifier/CNN classes

        Wraps .forward(), selecting embedding and possibly logits as outputs

        Note: when return_preds=True, values returned depend on self.use_custom_classifier:
            - if True, returns outputs of self.network()
            - if False, returns original Perch2 class logits

        Args:
            samples: same as CNN.predict(): list of file paths, OR pd.DataFrame with index
                containing audio file paths, OR a pd.DataFrame with multi-index (file, start_time,
                end_time)
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            return_preds: bool, if True, returns two outputs (embeddings, logits)
                - logits are Perch2 class logits if self.use_custom_classifier=False,
                and self.network() outputs if self.use_custom_classifier=True
            return_dfs: bool, if True, returns embeddings as pd.DataFrame with multi-index like
                .predict(). if False, returns np.array of embeddings [default: True].
            wandb_session: [default: None] a weights and biases session object
            **dataloader_kwargs are passed to self.predict_dataloader()

        Returns: (embeddings, preds) if return_preds=True or embeddings if return_preds=False
            types are pd.DataFrame if return_dfs=True, or np.array if return_dfs=False
            - preds are always original Perch2 class logit scores with no activation layer applied
            - embeddings are the feature vectors from the penultimate layer of the network
            If other outputs are desired, eg custom classifier logits, use forward()

        See also:
            - predict: get per-audio-clip species outputs as pandas DataFrame
            - forward: get all of Perch's outputs, or any subset
        """
        if return_preds:
            if self.use_custom_classifier:  # self.network outputs
                return_values = ("embeddings", "custom_classifier_logits")
            else:  # Perch2 original class logits
                return_values = ("embeddings", "logits")
        else:
            return_values = ("embeddings",)

        # run inference, getting embeddings and optionally logits
        outs = self.forward(
            samples=samples,
            progress_bar=progress_bar,
            wandb_session=wandb_session,
            return_values=return_values,
            return_dfs=return_dfs,
            **dataloader_kwargs,
        )

        if return_preds:
            if self.use_custom_classifier:
                return outs["embeddings"], outs["custom_classifier_logits"]
            else:
                return outs["embeddings"], outs["logits"]
        else:
            return outs["embeddings"]
