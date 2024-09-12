from pathlib import Path

import pandas as pd
import numpy as np
import torch

import opensoundscape
from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import CNN
from opensoundscape import Audio, Action

from bioacoustics_model_zoo.utils import (
    AudioSampleArrayDataloader,
    download_github_file,
)


class MLPClassifier(torch.nn.Module):
    """initialize a fully connected NN with ReLU activations"""

    def __init__(self, input_size, output_size, hidden_layer_sizes=()):
        super().__init__()
        # constructor_name tuple informs how to recreate the network shape
        self.constructor_name = (input_size, output_size, hidden_layer_sizes)
        self.add_module("hidden_layers", torch.nn.Sequential())
        shapes = [input_size] + list(hidden_layer_sizes) + [output_size]
        for i, (in_size, out_size) in enumerate(zip(shapes[:-2], shapes[1:-1])):
            self.hidden_layers.add_module(
                f"layer_{i}", torch.nn.Linear(in_size, out_size)
            )
            self.hidden_layers.add_module(f"relu_{i}", torch.nn.ReLU())
        self.add_module("classifier", torch.nn.Linear(shapes[-2], shapes[-1]))
        self.classifier_layer = "classifier"

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x


class BirdNET(CNN):
    def __init__(
        self,
        checkpoint_url="https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
        label_url="https://github.com/kahst/BirdNET-Analyzer/blob/main/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_af.txt",
        num_tflite_threads=1,
    ):
        """load BirdNET model from .tflite file on GitHub

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
            url: url to .tflite checkpoint on GitHub, or a local path to the .tflite file
            label_url: url to .txt file with class labels, or a local path to the .txt file

        Returns:
            model object with methods for generating predictions and embeddings

        Methods:
            predict: get per-audio-clip per-class scores in dataframe format; includes WandB logging
                (inherited from BaseClassifier)
            embed: make embeddings for audio data (feature vectors from penultimate layer)

        Example:
        ```
        import torch
        m=torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'BirdNET',trust_repo=True)
        m.predict(['test.wav'],batch_size=64) # returns dataframe of per-class scores
        m.embed(['test.wav']) # returns dataframe of embeddings
        ```
        """
        # only require tensorflow if/when this class is used
        try:
            from tensorflow import lite as tflite
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "BirdNet requires tensorflow package to be installed. "
                "Install in your python environment with `pip install tensorflow`"
            ) from exc

        # load class list:
        if label_url.startswith("http"):
            label_path = download_github_file(label_url)
        else:
            label_path = label_url
        label_path = Path(label_path).resolve()  # get absolute path
        assert label_path.exists(), f"Label path {label_path} does not exist"

        # labels.txt is a single column of class names without a header
        classes = pd.read_csv(label_path, header=None)[0].values

        # initialize parent class with some default values
        # default custom classifier is one fully connected layer
        # user can create a different classifier before training
        # clf is assigned to .network and is the only part that is trained with .trai(),
        # the birdnet feature extractor (self.tf_model) is frozen
        # note that this clf will have random weights
        self.embedding_size = 1024
        clf = MLPClassifier(input_size=self.embedding_size, output_size=len(classes))
        super().__init__(architecture=clf, classes=classes, sample_duration=3)

        self.use_custom_classifier = False
        """(bool) whether to generate predictions using BirdNET or the custom classifier

        initially False, but gets set to True if .train() is called
        """

        # download model if URL, otherwise find it at local path:
        if checkpoint_url.startswith("http"):
            print("downloading model from URL...")
            model_path = download_github_file(checkpoint_url)
        else:
            model_path = checkpoint_url

        model_path = str(Path(model_path).resolve())  # get absolute path as string
        assert Path(model_path).exists(), f"Model path {model_path} does not exist"

        # load tflite model
        self.tf_model = tflite.Interpreter(
            model_path=model_path, num_threads=num_tflite_threads
        )
        self.tf_model.allocate_tensors()

        # initialize preprocessor and choose dataloader class
        self.preprocessor = AudioPreprocessor(sample_duration=3, sample_rate=48000)
        # extend short samples to 3s by padding end with zeros (silence)
        self.preprocessor.insert_action(
            action_index="extend",
            action=Action(Audio.extend_to, is_augmentation=False, duration=3),
        )
        self.inference_dataloader_cls = AudioSampleArrayDataloader
        self.train_dataloader_cls = AudioSampleArrayDataloader

    def initialize_custom_classifier(self, hidden_layer_sizes=(), classes=None):
        """initialize a custom classifier to replace the BirdNET classifier head

        The classifier is a multi-layer perceptron with ReLU activations and a final
        linear layer. The input size is the size of the embeddings from the BirdNET model,
        and the output size is the number of classes.

        The new classifier (self.network) will have random weights, and can be trained with
        self.train().

        Args:
            hidden_layer_sizes: tuple of int, sizes of hidden layers in the classifier
                [default: ()]
            classes: list of str, class names for the classifier. Will run self.change_classes(classes)
                if not None. If None, uses self.classes. [default: None]

        Effects:
            sets self.network to a new MLPClassifier with the specified shape
            runs self.change_classes(classes) if classes is not None, resulting in
                self.classes = classes
        """
        if classes is not None:
            self.change_classes(classes)
        self.network = MLPClassifier(
            input_size=self.embedding_size,
            output_size=len(self.classes),
            hidden_layer_sizes=hidden_layer_sizes,
        )

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

    def training_step(self, samples, batch_idx):
        """a standard Lightning method used within the training loop, acting on each batch

        returns loss

        Effects:
            logs metrics and loss to the current logger
        """
        # switch from BirdNET classifier head to custom classifier (self.network)
        self.use_custom_classifier = True
        batch_data, batch_labels = samples
        # first run the preprocessed samples through the birdnet feature extractor
        # discard the logits produced by the birdnet classifier, just keep embeddings
        embeddings, _ = self._batch_forward(batch_data)
        # then run the embeddings through the trainable classifier with a typical "train" step
        return super().training_step(
            (torch.tensor(embeddings), torch.tensor(batch_labels)), batch_idx
        )

    def save(self, *args, **kwargs):
        """save model to path

        Note: the tf Interpreter is not pickable, so it is set to None before saving
        and re-assigned after saving (even if saving fails)

        Args: see opensoundscape.ml.cnn.CNN.save()
        """
        assert isinstance(
            self.network, MLPClassifier
        ), "model.save() and model.load() only supports reloading .network if it is and instance of the Classifier class"

        # since the tf Interpreter is not pickable, we don't include it in the saved file
        # instead we can recreate it with __init__ and a checkpoint
        temp_tf_model = self.tf_model
        try:
            # save the model with .tf_model (Tensorflow Interpreter) set to None
            self.tf_model = None
            super().save(*args, **kwargs)
        finally:
            # reassign the Tensorflow Interpreter to .tf_model
            # this needs to happen even if the model saving fails!
            self.tf_model = temp_tf_model

    @classmethod
    def load(cls, path, **kwargs):
        """load model from path

        re-loads custom classifier head trained by user
        (only supports the Classifier class)

        Sets self.use_custom_classifier to True, since loading from a file
        is only necessary if the user has trained a custom classifier
        (otherwise simply use BirdNET() to create a new model)

        Args:
            path: path to model saved using BirdNET.save()
            kwargs are passed to self.__init__()

        Returns:
            model including any custom trained classifier head
        """
        import warnings

        model_dict = torch.load(path)

        opso_version = (
            model_dict.pop("opensoundscape_version")
            if isinstance(model_dict, dict)
            else model_dict.opensoundscape_version
        )
        if opso_version != opensoundscape.__version__:
            warnings.warn(
                f"Model was saved with OpenSoundscape version {opso_version}, "
                f"but you are currently using version {opensoundscape.__version__}. "
                "This might not be an issue but you should confirm that the model behaves as expected."
            )

        if isinstance(model_dict, dict):
            # load up the weights and instantiate from dictionary keys
            # includes preprocessing parameters and settings
            # assumes that model_dict["architecture"] is a tuple of args for __init__
            # of the MLPClassifier class (which will be true if model was saved with BirdNET.save()
            # and MLPClassifier was used to create the model.network)
            model = cls(**kwargs)
            hidden_layer_sizes = model_dict["architecture"][2]
            model.initialize_custom_classifier(
                hidden_layer_sizes=hidden_layer_sizes, classes=model_dict["classes"]
            )
            # load state dict of custom classifier
            model.network.load_state_dict(model_dict["weights"])

        else:
            model = model_dict  # entire pickled object, not dictionary
            opso_version = model.opensoundscape_version

            # create the BirdNET TF Interpreter using __init__ with default args
            # since it is not saved with self.save() due to pickling issues
            model.tf_model = cls(**kwargs).tf_model

        model.use_custom_classifier = True
        return model
