from pathlib import Path

import torch
import pandas as pd
import numpy as np

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier

from bioacoustics_model_zoo.utils import (
    collate_to_np_array,
    AudioSampleArrayDataloader,
    download_github_file,
)


class BirdNET(BaseClassifier):
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

        Args:
            url: url to .tflite checkpoint on GitHub, or a local path to the .tflite file
            label_url: url to .txt file with class labels, or a local path to the .txt file

        Returns:
            model object with methods for generating predictions and embeddings

        Methods:
            predict: get per-audio-clip per-class scores in dataframe format; includes WandB logging
                (inherited from BaseClassifier)
            generate_embeddings: make embeddings for audio data (feature vectors from penultimate layer)
            generate_embeddings_and_logits: returns (embeddings, logits)


        Example:
        ```
        import torch
        m=torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'BirdNET',trust_repo=True)
        m.predict(['test.wav']) # returns dataframe of per-class scores
        m.generate_embeddings(['test.wav']) # returns dataframe of embeddings
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
        df_index = dataloader.dataset.dataset.label_df.index
        embeddings = self(dataloader, return_embeddings=True, return_logits=False)
        return pd.DataFrame(index=df_index, data=embeddings)

    def generate_embeddings_and_logits(self, samples, **kwargs):
        """Return (logits, embeddings) dataframes for audio data

        avoids running inference twice, so faster than calling
        generate_embeddings and generate_logits separately

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        embeddings, logits = self(
            dataloader, return_embeddings=True, return_logits=True
        )
        df_index = dataloader.dataset.dataset.label_df.index
        return (
            pd.DataFrame(index=df_index, data=embeddings),
            pd.DataFrame(index=df_index, data=logits, columns=self.classes),
        )
