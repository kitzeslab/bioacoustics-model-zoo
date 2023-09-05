from pathlib import Path

import torch
import pandas as pd
import numpy as np

# copying this import logit from BirdNet-Analyzer
# try:
#     import tflite_runtime.interpreter as tflite
# except ModuleNotFoundError:
from tensorflow import lite as tflite

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier

from bioacoustics_model_zoo.utils import (
    collate_to_np_array,
    AudioSampleArrayDataloader,
    download_github_file,
)

# Birdnet Analyzer provides good api: https://github.com/kahst/BirdNET-Analyzer/blob/main/model.py


def birdnet(
    checkpoint_url="https://github.com/kahst/BirdNET-Analyzer/blob/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
    label_url="https://github.com/kahst/BirdNET-Analyzer/blob/main/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_af.txt",
):
    return BirdNetTFLite(checkpoint_url, label_url)


class BirdNetTFLite(BaseClassifier):
    """load BirdNET model from .tflite file (does url work?)

    Args:
        url to model path (default is v2.4 FP16)

    Returns:
        opensoundscape.TensorFlowHubModel object with .predict() method for inference

    Methods:
        predict: get per-audio-clip per-class scores in dataframe format; includes WandB logging
        generate_embeddings: make embeddings for audio data (feature vectors from penultimate layer)
        generate_embeddings_and_logits: returns (embeddings, logits)
    """

    def __init__(
        self,
        checkpoint_url="https://github.com/kahst/BirdNET-Analyzer/raw/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
        label_url="https://github.com/kahst/BirdNET-Analyzer/blob/main/labels/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels_af.txt",
        num_tflite_threads=1,
    ):
        """load model, wrap in OpSo class

        Args:
            url: url to .tflite checkpoint on GitHub, or a local path to the .tflite file
            label_url: url to .txt file with class labels, or a local path to the .txt file

        Returns:
            object with .predict() method for inference
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
        self.network = tflite.Interpreter(
            model_path=model_path, num_threads=num_tflite_threads
        )
        self.network.allocate_tensors()

        # load class list:
        if label_url.startswith("http"):
            label_path = download_github_file(label_url)
        else:
            label_path = label_url
        label_path = Path(label_path).resolve()  # get absolute path
        assert label_path.exists(), f"Label path {label_path} does not exist"

        # labels.txt is a single column of class names without a header
        self.classes = pd.read_csv(label_path, header=None)[0].values

        # initialize preprocessor and choose dataloader class
        self.preprocessor = AudioPreprocessor(sample_duration=3, sample_rate=48000)
        self.inference_dataloader_cls = AudioSampleArrayDataloader

    def __call__(
        self, dataloader, return_embeddings=False, return_logits=True, **kwargs
    ):
        """kwargs are passed to SafeAudioDataloader init (num_workers, batch_size, etc)"""

        if not return_logits and not return_embeddings:
            raise ValueError("Both return_logits and return_embeddings cannot be False")

        input_details = self.network.get_input_details()[0]
        output_details = self.network.get_output_details()[0]
        embedding_idx = output_details["index"] - 1

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        for batch in tqdm(dataloader):
            for audio in batch:  # no batching, one by one?
                # using chirp repo code here:
                self.network.set_tensor(
                    input_details["index"], np.float32(audio)[np.newaxis, :]
                )
                self.network.invoke()
                logits.extend(self.network.get_tensor(output_details["index"]))
                embeddings.extend(self.network.get_tensor(embedding_idx))

        if return_logits and return_embeddings:
            return embeddings, logits
        elif return_logits:
            return logits
        elif return_embeddings:
            return embeddings

    def generate_embeddings(self, samples, **kwargs):
        """Generate embeddings for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        Returns:
            list of embeddings
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=True, return_logits=False)

    def generate_logits(self, samples, **kwargs):
        """Return (logits, embeddings) for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=False, return_logits=True)

    def generate_embeddings_and_logits(self, samples, **kwargs):
        """Return (logits, embeddings) for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        return self(dataloader, return_embeddings=True, return_logits=True)
