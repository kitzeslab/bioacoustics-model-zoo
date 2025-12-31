from pathlib import Path

import pandas as pd
import numpy as np
import torch

import opensoundscape
from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import CNN
from opensoundscape import Audio, Action

from bioacoustics_model_zoo.utils import (
    AudioSampleArrayDataloader,
    download_file,
    download_cached_file,
    register_bmz_model,
)
from bioacoustics_model_zoo.tensorflow_wrapper import (
    TensorFlowModelWithPytorchClassifier,
)


@register_bmz_model
class BirdNET(TensorFlowModelWithPytorchClassifier):
    def __init__(
        self,
        checkpoint_url="https://github.com/kahst/BirdNET-Analyzer/blob/v1.3.1/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
        label_url="https://github.com/kahst/BirdNET-Analyzer/blob/v1.3.1/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_af.txt",
        num_tflite_threads=1,
        cache_dir=None,
        version="2.4",
    ):
        """load BirdNET global bird classification CNN from .tflite file on GitHub

        [BirdNET](https://github.com/kahst/BirdNET-Analyzer) is shared under the CC A-NC-SA 4.0.
        Suggested Citation:
        @article{kahl2021birdnet,
            title={BirdNET: A deep learning solution for avian diversity monitoring},
            author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
            journal={Ecological Informatics},
            volume={61},
            pages={101236},
            year={2021},
            publisher={Elsevier}
        }

        BirdNET Analyzer provides good api: https://github.com/kahst/BirdNET-Analyzer/blob/main/model.py
        This wrapper may be useful for those already using OpenSoundscape and looking for a consistent API

        Dependencies:

        Tensorflow can be finicky about compatible versions of packages. This combination of packages works:
        ```
        tensorflow==2.14.0
        tensorflow-estimator==2.14.0
        tensorflow-hub==0.14.0
        tensorflow-io-gcs-filesystem==0.34.0
        tensorflow-macos==2.14.0
        ```

        Args:
            checkpoint_url: url to .tflite checkpoint on GitHub, or a local path to the .tflite file
            label_url: url to .txt file with class labels, or a local path to the .txt file
            num_tflite_threads: number of threads for TFLite interpreter
            cache_dir: directory to cache downloaded files (uses default cache if None)
            version: only '2.4' is currently supported for automatic download. However,
                if you are specifying checkpoint downloads or local paths for version other than
                2.4, pass the BirdNet model version to correctly set the model.version attribute
                (used for model caching).

        Returns:
            model object with methods for generating predictions and embeddings

        Methods:
            predict: get per-audio-clip per-class scores in dataframe format; includes WandB logging
                (inherited from opensoundscape.SpectrogramClassifier)
            embed: make embeddings for audio data (feature vectors from penultimate layer)

        Example:
        ```
        import bioacoustics_model_zoo as bmz
        m=bmz.BirdNET()
        m.predict(['test.wav'],batch_size=64) # returns dataframe of per-class scores
        m.embed(['test.wav']) # returns dataframe of embeddings
        ```
        """
        self.version = str(version)

        # only require tensorflow if/when this class is used
        try:
            import ai_edge_litert.interpreter as tflite

            resolver = tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        except ModuleNotFoundError as exc:
            try:
                from tensorflow import lite as tflite

                resolver = tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "BirdNet requires tensorflow package to be installed. "
                    "Install in your python environment with `pip install tensorflow`"
                ) from exc

        # load class list:
        if label_url.startswith("http"):
            label_path = download_cached_file(
                label_url,
                filename=None,
                model_name="birdnet",
                model_version=self.version,
                cache_dir=cache_dir,
            )
        else:
            label_path = label_url
        label_path = Path(label_path).resolve()  # get absolute path
        assert label_path.exists(), f"Label path {label_path} does not exist"

        # labels.txt is a single column of class names without a header
        classes = pd.read_csv(label_path, header=None)[0].values

        # initialize parent class with some default values
        # default custom classifier is one fully connected layer
        # user can create a different classifier before training
        # clf is assigned to .network and is the only part that is trained with .train(),
        # the birdnet feature extractor (self.tf_model) is frozen
        # note that this clf will have random weights
        self.embedding_size = 1024
        sample_duration = 3  # fixed input size 3 seconds for birdnet
        super().__init__(
            embedding_size=self.embedding_size,
            classes=classes,
            sample_duration=sample_duration,
        )

        # download model if URL, otherwise find it at local path:
        if checkpoint_url.startswith("http"):
            print("downloading model from URL...")
            model_path = download_cached_file(
                checkpoint_url,
                filename=None,
                model_name="birdnet",
                model_version=self.version,
                cache_dir=cache_dir,
            )
        else:
            model_path = checkpoint_url

        model_path = str(Path(model_path).resolve())  # get absolute path as string
        assert Path(model_path).exists(), f"Model path {model_path} does not exist"

        self.tf_model = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_tflite_threads,
            experimental_op_resolver_type=resolver,
        )
        self.tf_model.allocate_tensors()

        # initialize preprocessor and choose dataloader class
        self.preprocessor = AudioAugmentationPreprocessor(
            sample_duration=sample_duration, sample_rate=48000
        )
        # extend short samples to 3s by padding end with zeros (silence)
        self.preprocessor.insert_action(
            action_index="extend",
            action=Action(
                Audio.extend_to, is_augmentation=False, duration=sample_duration
            ),
        )
        self.inference_dataloader_cls = AudioSampleArrayDataloader
        self.train_dataloader_cls = AudioSampleArrayDataloader

    def batch_forward(self, batch_data, targets=(-1,), avgpool=True):
        """run forward pass on a batch of data

        Args:
            batch_data: np.array of shape [batch, ...]
            targets: tuple of str, which outputs to return:
                - use -1 to get logits from BirdNET final layer
                - use "custom_classifier_logits" to get logits from custom classifier head (self.network)
            avgpool: not implemented (BirdNET embeddings are already pooled)

        Returns: dictionary of outputs for each key in targets
            Note: the classification logits (-1 key) depend on `self.use_custom_classifier`:
            - if False, the final layer of the BirdNET model is used
            - if True, the custom classifier (self.network) is used
        """
        input_details = self.tf_model.get_input_details()[0]
        input_layer_idx = input_details["index"]
        output_details = self.tf_model.get_output_details()[0]

        # choose which layer should be used for embeddings
        embedding_idx = output_details["index"] - 1

        # we need to reshape the format of expected input tensor for TF model to include batch dimension
        self.tf_model.resize_tensor_input(
            input_layer_idx, [len(batch_data), *batch_data[0].shape]
        )
        self.tf_model.allocate_tensors()  # memory allocation?
        # send data to model
        self.tf_model.set_tensor(input_details["index"], np.float32(batch_data))
        self.tf_model.invoke()  # forward pass

        outs = {}
        batch_embeddings = self.tf_model.get_tensor(embedding_idx)
        if "embedding" in targets:
            if "embedding" in targets:
                outs["embedding"] = batch_embeddings
        if -1 in targets:  # add class logit score predictions
            if self.use_custom_classifier:
                # run self.network on the features from birdnet to predict using self.network
                tensors = torch.tensor(batch_embeddings).to(self.device)
                self.network.to(self.device)
                outs[-1] = self.network(tensors).detach().cpu().numpy()
            else:
                outs[-1] = self.tf_model.get_tensor(output_details["index"])

        return outs

    def _check_or_get_default_embedding_layer(self, target_layer=None):
        return "embedding"


@register_bmz_model
class BirdNETOccurrenceModel:
    """Predict probability of bird species at week, lat, lon

    based on eBird checklist data

    adapted from BirdNET-Analyzer occurrence meta-model on GitHub

    Args:
        checkpoint_url: url to .tflite checkpoint on GitHub, or a local path to the .tflite file
        label_url: url to .txt file with class labels, or a local path to the .txt file
        num_tflite_threads: number of threads for TFLite interpreter
        cache_dir: directory to cache downloaded files (uses default cache if None)

    Returns:
        model object with methods for generating species probabilities and species lists

    Example:

    ```python
    from bioacoustics_model_zoo import BirdNETOccurrenceModel
    m=BirdNETOccurrenceModel()
    m.get_species_list(lat=37.419871, lon=-119.153168,week=20,threshold=.1)
    ```
    """

    def __init__(
        self,
        checkpoint_url="https://github.com/kahst/BirdNET-Analyzer/blob/v1.3.1/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite",
        label_url="https://github.com/kahst/BirdNET-Analyzer/blob/v1.3.1/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_af.txt",
        num_tflite_threads=1,
        cache_dir=None,
    ):
        # only require tensorflow if/when this class is used
        try:
            import ai_edge_litert.interpreter as tflite

            resolver = tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        except ModuleNotFoundError as exc:
            try:
                from tensorflow import lite as tflite

                resolver = tflite.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "BirdNet requires tensorflow package to be installed. "
                    "Install in your python environment with `pip install tensorflow`"
                ) from exc

        self.version = "2.4"

        # load class list:
        if label_url.startswith("http"):
            label_path = download_cached_file(
                label_url,
                filename=None,
                model_name="birdnet",
                model_version=self.version,
                cache_dir=cache_dir,
            )
        else:
            label_path = label_url
        label_path = Path(label_path).resolve()  # get absolute path
        assert label_path.exists(), f"Label path {label_path} does not exist"

        # labels.txt is a single column of class names without a header
        self.classes = pd.read_csv(label_path, header=None)[0].values

        self.scientific_names = [c.split("_")[0] for c in self.classes]
        self.common_names = [c.split("_")[1] for c in self.classes]

        # download model if URL, otherwise find it at local path:
        if checkpoint_url.startswith("http"):
            print("downloading model from URL...")
            model_path = download_cached_file(
                checkpoint_url,
                filename=None,
                model_name="birdnet",
                model_version=self.version,
                cache_dir=cache_dir,
            )
        else:
            model_path = checkpoint_url

        model_path = str(Path(model_path).resolve())  # get absolute path as string
        assert Path(model_path).exists(), f"Model path {model_path} does not exist"

        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_tflite_threads,
            # XNNPACK disabled, because it does not support variable inputsize anyway (ie batchsize)
            experimental_op_resolver_type=resolver,
        )
        self.interpreter.allocate_tensors()
        self.input_layer_index = self.interpreter.get_input_details()[0]["index"]
        self.output_layer_index = self.interpreter.get_output_details()[0]["index"]

    def species_probabilities(self, lat, lon, week):
        """Predicts the probability for each species at lat/lon/week.

        Args:
            lat: The latitude.
            lon: The longitude.
            week: The week of the year [1-48]. Use -1 for yearlong.

        Returns:
            A list of probabilities for all species.
        """

        # Prepare mdata as sample
        sample = np.expand_dims(np.array([lat, lon, week], dtype="float32"), 0)

        # Run inference
        self.interpreter.set_tensor(self.input_layer_index, sample)
        self.interpreter.invoke()

        return self.interpreter.get_tensor(self.output_layer_index)[0]

    def get_species_list(
        self,
        lat=-1,
        lon=-1,
        week=-1,
        threshold=0.03,
    ):
        """Predicts the species list at the coordinates and week of year, based on occurrence threshold

        Predicts the species list based on the coordinates and week of year.

        Note that the probability is relative to the most common species rather than absolute occurrence rate.

        Args:
            lat (float, optional): Latitude of the location for species filtering. Defaults to -1 (no filtering by location).
            lon (float, optional): Longitude of the location for species filtering. Defaults to -1 (no filtering by location).
            week (int, optional): Week of the year for species filtering. Defaults to -1 (no filtering by time).
            threshold (float, optional): Species frequency threshold for filtering. Defaults to 0.03.

        Returns:
            A sorted dataframe with columns: scientific_name, common_name, probability
        """
        # Make species probability prediction
        species_probs = self.species_probabilities(lat, lon, week)

        species_df = pd.DataFrame(
            {
                "scientific_name": self.scientific_names,
                "common_name": self.common_names,
                "relative_probability": species_probs,
            }
        )

        # Apply threshold
        filtered_df = species_df[species_df["relative_probability"] >= threshold]

        # Sort by relative probability and return
        return filtered_df.sort_values(
            by="relative_probability", ascending=False
        ).reset_index(drop=True)
