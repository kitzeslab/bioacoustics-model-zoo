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
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from opensoundscape.ml.cnn import load_or_create_hoplite_usearch_db
import opensoundscape as opso
from opensoundscape.sample import collate_audio_samples

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

    def embed_to_hoplite_db(
        self,
        samples,
        db,
        dataset_name,
        progress_bar=True,
        audio_root=None,
        embedding_exists_mode="skip",  # skip, error, add
        commit_frequency_batches=1,
        overflow_mode="warn",
        **dataloader_kwargs,
    ):
        """Run inference on a dataloader, saving 1D outputs of target_layer to a hoplite database

        Args:
            samples: (same as CNN.predict())
            db: a hoplite database object or a path to a hoplite database folder
                - if a path is provided, the database will be created if it does not exist
                - when creating a new db, the embedding_dim argument must be provided
            dataset_name: name of the dataset to save embeddings within
                - if the dataset does not exist in the db, it will be created
                - one hoplite database can contain multiple datasets
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            audio_root: the root directory for relative paths to audio files
            embedding_exists_mode: str, behavior when an embedding already exists for a given
                (dataset_name, source_id, offset) tuple. Options are:
                    "skip": skip inserting the embedding (default)
                    "error": raise an error
                    "add": add a new embedding entry to the db with the same source info
                Note that hoplite doesn't currently support removing or replacing existing entries
            commit_frequency_batches: int, commit to db after every N batches[default: 1]
            overflow_mode: 'warn', 'error', or 'ignore' behvior when embedding values exceed
                the range of float16, which is the range of values allowed in hoplite db
            embedding_dim: int, dimension of the embeddings to be stored
                - only used when creating a new hoplite db
                - must match the output dimension of the model's target_layer
                - if creating new db and embedding_dim is None, guesses based on self.classifier.in_features
            **dataloader_kwargs: additional keyword arguments to pass to the dataloader

        Returns:
            (embedding_db, dict with info about failed samples)
        """
        try:
            import perch_hoplite
            from perch_hoplite.db import interface as hoplite_interface
        except ImportError as e:
            raise ImportError(
                "hoplite is not installed. Please install hoplite to use this feature."
            ) from e

        # potentially: store db.metadata.dataset_paths -> dataset_name:audio_root mapping in db
        # and warn user if overwriting an existing mapping with a different audio_root

        # load or create hoplite db if a path is provided (if hoplite db is passed, use it directly)
        db = load_or_create_hoplite_usearch_db(db, embedding_dim=self.embedding_size)

        # create dataloader, collate using `identity` to return list of AudioSample
        # rather than (samples, labels) tensors
        dataloader = self.predict_dataloader(
            samples,
            collate_fn=opso.utils.identity,
            audio_root=audio_root,
            **dataloader_kwargs,
        )

        if embedding_exists_mode in ["skip", "error"]:
            # filter dataloader to only samples that don't already have embeddings
            # dataloader.dataset/
            index_values = list(dataloader.dataset.dataset.label_df.index)

            # check if each exists
            keep_idxs = []
            for i, (file, start_time, _) in enumerate(index_values):
                matching_ids = db.get_embeddings_by_source(
                    dataset_name=dataset_name,
                    source_id=str(file),
                    offsets=np.array([start_time], dtype=np.float16),
                )
                if len(matching_ids) == 0:
                    keep_idxs.append(i)
                elif embedding_exists_mode == "error":
                    # don't allow adding or skipping duplicated entries
                    raise ValueError(
                        f"Embedding already exists for {file}:{start_time} in dataset {dataset_name}"
                        " and embedding_exists_mode='error'. Other options are 'skip' or 'add'. "
                    )

            # subset the label_df to exclude duplicated_idxs
            dataloader.dataset.dataset.label_df = (
                dataloader.dataset.dataset.label_df.iloc[keep_idxs]
            )
            new_len = len(dataloader.dataset.dataset.label_df)

            if new_len == 0:
                # all samples already have embeddings, nothing to do
                print("all samples already have embeddings in the database")
                return db, {}
        # elif embedding_exists_mode == "add": # do nothing, allow duplicates

        # disable gradient updates during inference
        for i, batch_samples in enumerate(tqdm(dataloader, disable=not progress_bar)):

            batch_tensors, _ = collate_audio_samples(batch_samples)
            batch_tensors.requires_grad = False
            model_outputs = self.tf_model.signatures["serving_default"](
                inputs=batch_tensors.numpy()
            )

            # move tensorflow tensors to CPU and convert to numpy
            outs = {
                k: None if v is None else v.numpy() for k, v in model_outputs.items()
            }

            batch_emb = outs["embedding"]

            # insert the embeddings one-by-one to the hoplite db

            max_float16 = np.finfo(np.float16).max
            # we clip values to the float16 range before casting, to avoid overfloat -> inf values
            if np.abs(batch_emb).max() > max_float16:
                if overflow_mode == "warn":
                    warnings.warn("clipping embedding values to float16 range")
                elif overflow_mode == "error":
                    raise ValueError("Embeddings exceeded float16 range")
                # otherwise clip without warnings/errors
            batch_emb = batch_emb.clip(-max_float16, max_float16).astype(np.float16)

            # insert each embedding in the batch into the database, one-by-one
            for j in range(batch_tensors.shape[0]):
                file = batch_samples[j].source
                if audio_root is not None:
                    # use the relative path for the source name stored in the db
                    file = str(Path(file).relative_to(audio_root))
                start_time = batch_samples[j].start_time

                emb_source = hoplite_interface.EmbeddingSource(
                    dataset_name=dataset_name,
                    source_id=file,
                    offsets=np.array([start_time], np.float16),
                )
                db.insert_embedding(batch_emb[j], emb_source)

            # commit once per commit_frequency_batches batches
            # committing is relatively slow, but we don't want to lose progress if interrupted
            if (i + 1) % commit_frequency_batches == 0:
                db.commit()

        # end of batch loop

        # commit any remaining embeddings!
        db.commit()

        # return database object and info about any failed samples
        return (db, {"failed_samples": dataloader.dataset.report()})

    def similarity_search_hoplite_db(
        self,
        query_samples,
        db,
        num_results=5,
        exact_search=False,
        search_subset_size=None,
        # datasets=None, # would like to implement filtering to specific datasets
        target_score=None,
        audio_root=None,
        search_kwargs=None,
        **embedding_kwargs,
    ):
        """Perform a similarity search in the Hoplite database.

        Args:
            query_samples: audio examples for which to find most similar examples
                file path, list of paths, or dataframe with `file,start_time,end_time` multi-index
            db: a Hoplite database containing embeddings from the same model
            num_results: The number of results to return for each query
            exact_search: default False for usearch (faster), if True uses brute force search
            search_subset_size: Number of embeddings to compare with. If None, all embeddings
                are used. For floats between 0 and 1, sample a proportion of the database.
                For ints, sample the specified number of embeddings.
                if None [default], searches all embeddings
                Note: only implemented for exact_search=True
            target_score: if specified, searches for similarity scores close to target_score
                default [None] searches for most similar embeddings
            audio_root: root directory for relative paths to query audio files
            search_kwargs: dict of additional keyword arguments passed to db.ui.search() or
                brutalism.threaded_brute_search() if exact_search=True
                exact_search=False: radius, threads, exact, log, progress
                exact_search=True: batch_size, max_workers, rng_seed
            **embedding_kwargs: additional keyword arguments passed to self.embed(), such as
                batch_size and num_workers
        Returns:
            A list of dictionaries with the search results, one item per query sample:
            Each item is a dictionary with the following keys:
                - "query": dictionary with query metadata
                - "results": list of dictionaries with metadata for each retrieved sample
        """
        try:
            from perch_hoplite.db import brutalism
            from perch_hoplite.db import score_functions
            from perch_hoplite.db import search_results
        except ImportError as e:
            raise ImportError(
                "hoplite is not installed. Please install hoplite to use this feature."
            ) from e

        if search_kwargs is None:
            search_kwargs = {}

        if not exact_search:
            if search_subset_size is not None:
                raise NotImplementedError(
                    "search_subset_size is only implemented for exact_search=True"
                )
            if target_score is not None:
                raise NotImplementedError(
                    "target_score is only implemented for exact_search=True"
                )

        # generate embeddings for the query samples
        print("embedding query samples")
        embeddings = self.embed(
            query_samples, audio_root=audio_root, **embedding_kwargs
        )

        print(
            f"performing similarity search for each of {embeddings.shape[0]} query samples"
        )
        compiled_search_results = []
        for (qfile, qstart_time, qend_time), emb in embeddings.iterrows():
            query_embedding = emb.values.astype(np.float16)
            if exact_search:
                score_fn = score_functions.get_score_fn(
                    "dot", target_score=target_score
                )
                results, all_scores = brutalism.threaded_brute_search(
                    db,
                    query_embedding,
                    num_results,
                    score_fn=score_fn,
                    sample_size=search_subset_size,
                    **search_kwargs,
                )

            else:
                ann_matches = db.ui.search(
                    query_embedding, count=num_results, **search_kwargs
                )
                results = search_results.TopKSearchResults(top_k=num_results)
                for k, d in zip(ann_matches.keys, ann_matches.distances):
                    results.update(search_results.SearchResult(k, d))

            # extract relevant info for each match into dictionaries
            results_list = []
            for match in results.search_results:
                clip_info = db.get_embedding_source(match.embedding_id)
                results_list.append(
                    {
                        "embedding_id": match.embedding_id,
                        "score": match.sort_score,
                        "dataset_name": clip_info.dataset_name,
                        "source_id": clip_info.source_id,
                        "offset": clip_info.offsets[0],
                    }
                )
            compiled_search_results.append(
                {
                    "query": {
                        "file": qfile,
                        "start_time": qstart_time,
                        "end_time": qend_time,
                        # "embedding": query_embedding,
                        "audio_root": audio_root,
                    },
                    "results": results_list,
                }
            )
        return compiled_search_results
