"""implements a class that wraps tensorflow compiled inference models with trainable pytorch classifier using opensoundscape

subclassed by BirdNET and Perch
"""

import pandas as pd
import torch

import opensoundscape
from opensoundscape.ml.cnn import CNN
from opensoundscape.ml.shallow_classifier import MLPClassifier


class TensorFlowModelWithPytorchClassifier(CNN):
    def __init__(self, embedding_size, classes, sample_duration):
        """ """
        # initialize a CNN where self.network is a pytorch classification head
        self.embedding_size = embedding_size
        clf = MLPClassifier(input_size=self.embedding_size, output_size=len(classes))
        super().__init__(
            architecture=clf, classes=classes, sample_duration=sample_duration
        )

        self.use_custom_classifier = False
        """(bool) whether to generate predictions using BirdNET or the custom classifier

        initially False, but gets set to True if .train() is called
        """

        self.tf_model = None  # set by subclass, holds the TensorFlow model

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

    @property
    def custom_classifier(self):
        """alias for self.network"""
        return self.network

    def _batch_forward(self, batch_data):
        """This method should return the tensorflow model output  logits if
        self.use_custom_classifier is False, otherwise it should use the custom
        classifier (self.network) to generate logits on the embeddings from the
        tensorflow model.

        It must return (embeddings, logits) on a single batch of samples
        """
        raise NotImplementedError("This method should be implemented in subclasses")

    def __call__(self, dataloader):
        """This method should call _batch_forward() for each batch in the dataloader

        Return values might depend on the model or be configurable through arguments
        """
        raise NotImplementedError("This method should be implemented in subclasses")

    def embed(self, samples):
        """This method should call __call__() on the output of self.predict_dataloader(samples)

        Return values might depend on the model or be configurable through arguments
        """
        raise NotImplementedError("This method should be implemented in subclasses")

    def training_step(self, samples, batch_idx):
        """a standard Lightning method used within the training loop, acting on each batch

        returns loss

        Effects:
            logs metrics and loss to the current logger
        """
        batch_data, batch_labels = samples
        # first run the preprocessed samples through the tensorflow feature extractor
        # discard the logits produced by the tensorflow classifier, just keep embeddings
        self.use_custom_classifier = False
        embeddings, _ = self._batch_forward(batch_data)
        # switch from Tensorflow classifier head to custom classifier (self.network)
        self.use_custom_classifier = True
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
        ), "model.save() and model.load() only supports reloading .network if it is and instance of the MLPClassifier class"

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

            # re-create the tensorflow inference model using __init__
            # since it is not saved with self.save() due to pickling issues
            model.tf_model = cls(**kwargs).tf_model

        model.use_custom_classifier = True
        return model
