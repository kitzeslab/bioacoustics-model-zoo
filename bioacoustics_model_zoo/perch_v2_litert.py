from bioacoustics_model_zoo.utils import register_bmz_model
from bioacoustics_model_zoo.tensorflow_wrapper import (
    TensorFlowModelWithPytorchClassifier,
)

import pandas as pd
import torch
import warnings

import opensoundscape
from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape import Action, Audio
from bioacoustics_model_zoo.utils import (
    AudioSampleArrayDataloader,
    register_bmz_model,
)
from pathlib import Path


@register_bmz_model
class Perch2LiteRT(TensorFlowModelWithPytorchClassifier):
    """load Perch v2.0 TFLite /Lite RT model

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
        model_path: str, path to local .tflite model file
        version: str or None, version identifier for the model
        num_tflite_threads: int, number of threads to use for tflite inference
            (default: 1)

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
    `pip install ai-edge-litert`
    """

    similarity_search_hoplite_db = (
        opensoundscape.ml.cnn.SpectrogramClassifier.similarity_search_hoplite_db
    )

    def __init__(self, model_path, version=None, num_tflite_threads=1):
        # only require litert
        try:
            from ai_edge_litert import interpreter
            from ai_edge_litert.interpreter import Interpreter

            # resolver = tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                """Perch2LiteRT requires the ai-edge-litert package.
                Please install it using:
                pip install ai-edge-litert
                """
            ) from exc

        # initialize parent class with methods for training custom classifier head
        super().__init__(
            embedding_size=1536,
            classes=list(range(14795)),  # class_lists["labels"]["classes"],
            sample_duration=5,
        )

        # which model to load depends on whether GPU is available
        model_path = str(Path(model_path).resolve())  # get absolute path as string
        assert Path(model_path).exists(), f"Model path {model_path} does not exist"

        # first try hub.load(url): succeeds to download but fails to find file within subfolder
        self.tf_model = Interpreter(
            model_path=model_path,
            num_threads=num_tflite_threads,
            # experimental_op_resolver_type=resolver,
        )
        self.tf_model.allocate_tensors()

        input_details = self.tf_model.get_input_details()
        signature_list = self.tf_model.get_signature_list()

        # input_shape = input_details[0]["shape"]
        self.tf_input_dtype = input_details[0]["dtype"]

        self.tf_inference_handle = self.tf_model.get_signature_runner("serving_default")
        self.tf_in_layer = signature_list["serving_default"]["inputs"][0]
        input_shape = input_details[0]["shape"]
        input_dtype = input_details[0]["dtype"]

        self.version = version
        # self.ebird_codes = class_lists["perch_v2_ebird_classes"]["classes"]
        self.inference_dataloader_cls = AudioSampleArrayDataloader
        self.train_dataloader_cls = AudioSampleArrayDataloader

        # Configure preprocessing
        # Perch expects audio signal input as 32kHz mono 5s clips (160,000 samples)
        self.sample_duration = 5
        self.preprocessor = AudioAugmentationPreprocessor(
            sample_duration=self.sample_duration, sample_rate=32000
        )

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

    def batch_forward(
        self,
        batch_data,
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
        # Run via signature
        # input_dtype = input_details[0]["dtype"]
        # sig = interpreter.get_signature_runner("serving_default")
        # result = sig(**{input_name: sample_input.astype(input_dtype)})

        model_outputs = self.tf_inference_handle(
            **{self.tf_in_layer: batch_data.astype(self.tf_input_dtype)}
        )

        # opensoundscape uses reserved key -1 for model outputs e.g. during .predict()
        if -1 in targets:
            model_outputs[-1] = model_outputs["label"]

        # subset results to only requested targets
        outs = {
            k: None if v is None else v
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
        return_values=("label", "embedding", "spatial_embedding", "spectrogram"),
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
            return_values: tuple(str,): select from 'label', 'embedding',
                'spatial_embedding', 'spectrogram', 'custom_classifier_logits'
                [default: ('label','embedding','spatial_embedding','spectrogram')]
                Include any combination of the following:
                - 'label': logit scores (class predictions) on the species classes
                - 'embedding': 1D feature vectors from the penultimate layer of the network
                - 'spatial_embedding': un-pooled spatial embeddings from the network
                - 'spectrogram': log-mel spectrograms generated during preprocessing
                - 'custom_classifier_logits': outputs of the custom classifier head (self.network)
            return_dfs: bool, if True, returns outputs as pd.DataFrame with multi-index like
                .predict() ('file','start_time','end_time'), if False, returns np.array
                [default: True]
            **dataloader_kwargs: additional keyword arguments passed to the dataloader such
                as batch_size, num_workers, etc.

        Returns: dictionary with content depending on return_values and return_dfs arguments:
            - 'label': pd.DataFrame or np.array of per-clip logits on species classes
                shape: (num_clips, num_classes)
            - 'embedding': pd.DataFrame or np.array of per-clip 1D feature vectors
                shape: (num_clips, 1536)
            - 'spatial_embedding': np.array of per-clip spatial embeddings
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
        # and appropriate column names
        if return_dfs:
            if "label" in results_dict:
                results_dict["label"] = pd.DataFrame(
                    data=results_dict["label"],
                    index=dataloader.dataset.dataset.label_df.index,
                    columns=self._original_classes,
                )
            if "embedding" in results_dict:
                results_dict["embedding"] = pd.DataFrame(
                    data=results_dict["embedding"],
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
