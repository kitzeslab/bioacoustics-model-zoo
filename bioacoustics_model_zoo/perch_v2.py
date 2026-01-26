from bioacoustics_model_zoo.utils import register_bmz_model
from bioacoustics_model_zoo.tensorflow_wrapper import (
    TensorFlowModelWithPytorchClassifier,
)

import pandas as pd
import torch
import warnings
import numpy as np

import opensoundscape
from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape import Action, Audio
from bioacoustics_model_zoo.utils import (
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

    Note: different model checkpoints are downloaded depending on whether `torch.cuda.is_available()`
    is True or False. If true, the default, GPU-only model is downloaded. If false, a CPU-compatible
    model is downloaded. The type of model loaded is indicated by `self.system` attribute.

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
        version: select from released versions of Perch v2.0 on Kaggle
            Default: None currently selects "2" for GPU-compatible model or
            "1" for CPU-compatible model (latest as of October 2025).

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

    Environment setup:

    Perch2 requires tensorflow >=2.20.0
    ```
    pip install --upgrade opensoundscape bioacoustics-model-zoo tensorflow tensorflow-hub
    ```
    """

    similarity_search_hoplite_db = (
        opensoundscape.ml.cnn.SpectrogramClassifier.similarity_search_hoplite_db
    )

    def __init__(self, version=None):
        # only require tensorflow and tensorflow_hub if/when this class is used
        try:
            import tensorflow_hub as hub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                """Perch2 requires tensorflow and tensorflow_hub packages >=2.20.0.
                Please install them using:
                pip install --upgrade opensoundscape bioacoustics-model-zoo tensorflow tensorflow-hub
                """
            ) from exc

        # which model to load depends on whether GPU is available
        if torch.cuda.is_available():
            system = "GPU"
            if version is None:
                version = 2  # latest GPU-compatible as of Oct 2025
            tested_versions = (2,)

            tfhub_path = f"https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/{version}"
        else:
            system = "CPU"
            if version is None:
                version = 1  # latest CPU-compatible as of Oct 2025
            tested_versions = (1,)
            tfhub_path = f"https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu/{version}"

        if not version in tested_versions:
            warnings.warn(
                f"version {version} has not been tested on {system}, tested versions: {tested_versions}"
            )
        self.version = version
        self.system = system

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

        # during inference, Perch rescales with per-sample peak normalization to 0.25
        # https://github.com/kitzeslab/bioacoustics-model-zoo/issues/30#issuecomment-3134186126
        self.preprocessor.insert_action(
            action_index="normalize_signal",
            action=Action(
                opensoundscape.Audio.normalize, is_augmentation=False, peak_level=0.25
            ),
        )

    def batch_forward(
        self,
        batch_samples,
        targets=("embedding", "spatial_embedding", "label", "spectrogram"),
        avgpool=False,
    ):
        """run inference on a single batch of samples

        Returns a dictionary of outputs for the each target

        Args:
            batch_data: np.array of audio samples, shape (batch_size, 32000*5)
            targets: tuple of str, select from
                ['embedding', 'spatial_embedding', 'label', 'spectrogram','custom_classifier_logits']
                - 'custom_classifier_logits' is the result of self.network() on the embeddings
            avgpool: ignored
        Returns:
            dict with keys matching targets, values are np.arrays of outputs
        """
        data = np.array([s.data.samples for s in batch_samples], dtype=np.float32)
        model_outputs = self.tf_model.signatures["serving_default"](inputs=data)

        # opensoundscape uses reserved key -1 for model outputs e.g. during .predict()
        if -1 in targets:
            model_outputs[-1] = model_outputs["label"]
        # move tensorflow tensors to CPU and convert to numpy
        # only retaining requested outputs
        outs = {
            k: None if v is None else v.numpy()
            for k, v in model_outputs.items()
            if k in targets
        }

        if "custom_classifier_logits" in targets:
            emb_tensor = torch.tensor(outs["embedding"]).to(self.device)
            self.network.to(self.device)
            outs["custom_classifier_logits"] = (
                self.network(emb_tensor).detach().cpu().numpy()
            )

        return outs

    def forward(
        self,
        samples,
        progress_bar=True,
        wandb_session=None,
        targets=("logits", "embeddings", "spatial_embeddings", "spectrograms"),
        return_dfs=True,
        **dataloader_kwargs,
    ):
        """
        Run inference on a list of samples, returning all selected outputs as a dictionary

        use "custom_classifier_logits" in return_values to get outputs from the
        custom classifier head (self.network)

        wraps self.predict_dataloader() and self.__call__() to run the model on
        audio files/clips, then optionally places 1D outputs in dataframes

        Args:
            samples: list of file paths, OR pd.DataFrame with index containing audio file paths
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            wandb_session: wandb.Session object, if provided, logs progress
            targets: tuple(str,): select from 'logits', 'embeddings',
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
            targets=targets,
        )

        # optionally put 1D outputs in DataFrames with multi-index ('file','start_time','end_time')
        # and appropriate column names
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

    def _check_or_get_default_embedding_layer(self, target_layer=None):
        """only allows 'embedding' or 'spatial_embedding' as target layers
        Args:
            target_layer: str or None, select from 'embedding' or 'spatial_embedding'
                If None, defaults to 'embedding

        Returns:
            str, the selected target layer
        """
        if target_layer == "spatial_embedding":
            return "spatial_embedding"
        else:
            return "embedding"
