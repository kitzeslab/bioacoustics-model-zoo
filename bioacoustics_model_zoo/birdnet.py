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
        super().__init__(embedding_size=1024, classes=classes, sample_duration=3)

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
            sample_duration=3, sample_rate=48000
        )
        # extend short samples to 3s by padding end with zeros (silence)
        self.preprocessor.insert_action(
            action_index="extend",
            action=Action(Audio.extend_to, is_augmentation=False, duration=3),
        )
        self.inference_dataloader_cls = AudioSampleArrayDataloader
        self.train_dataloader_cls = AudioSampleArrayDataloader

    def _batch_forward(self, batch_data):
        """run forward pass on a batch of data

        Args:
            batch_data: np.array of shape [batch, ...]

        Returns: (embeddings, logits)
            Note: the classification head used depends on `self.use_custom_classifier`:
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

        batch_embeddings = self.tf_model.get_tensor(embedding_idx)
        if self.use_custom_classifier:
            # run self.network on the features from birdnet to predict using self.network
            tensors = torch.tensor(batch_embeddings).to(self.device)
            self.network.to(self.device)
            batch_logits = self.network(tensors).detach().cpu().numpy()
        else:
            batch_logits = self.tf_model.get_tensor(output_details["index"])

        return batch_embeddings, batch_logits

    def __call__(
        self,
        dataloader,
        return_embeddings=False,
        return_logits=True,
        wandb_session=None,
        progress_bar=True,
    ):
        """forward pass

        Args:
            dataloader: dataloader object created with self.predict_dataloader(samples)
            return_embeddings, return_logits: bool, which outputs to return
                (cannot both be False; if both true, returns (logits, embeddings))
                [default: return_logits=True, return_embeddings=False]
            wandb_session: wandb.Session object, if provided, logs progress to wandb
            progress_bar: bool, if True, shows a progress bar with tqdm

        Returns: depends on return_embeddings and return_logits
            - if both True, returns (logits, embeddings)
            - if only return_logits, returns logits
            - if only return_embeddings, returns embeddings

            The classification head used depends on `self.use_custom_classifier`:
            - if False, the final layer of the BirdNET model is used
            - if True, the custom classifier (self.network) is used
        """
        if not return_logits and not return_embeddings:
            raise ValueError("Both return_logits and return_embeddings cannot be False")

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        for batch_data, _ in tqdm(dataloader, disable=not progress_bar):
            batch_embeddings, batch_logits = self._batch_forward(batch_data)
            if return_logits:
                logits.extend(batch_logits)
            if return_embeddings:
                embeddings.extend(batch_embeddings)

            if wandb_session is not None:
                wandb_session.log(
                    {
                        "progress": len(logits) / len(dataloader.dataset),
                        "completed_batches": len(logits),
                        "total_batches": len(dataloader.dataset),
                    }
                )
        logits = np.array(logits)
        embeddings = np.array(embeddings)

        if return_logits and return_embeddings:
            return logits, embeddings
        elif return_logits:
            return logits
        elif return_embeddings:
            return embeddings

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

        wraps self.__call__ by generating a dataloader and handling output preferences

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

        # run inference, returns (scores, embeddings)
        # or just (embeddings) if return_preds=False
        returns = self(
            dataloader=dataloader,
            progress_bar=progress_bar,
            return_embeddings=True,
            return_logits=return_preds,
        )
        # we got one output (embeddings) if return_preds is False or two outputs (logits, embeddings)
        # if return_preds is True
        if return_preds:
            preds, embeddings = returns
        else:
            embeddings = returns

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
