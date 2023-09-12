"""load the pre-trained YamNET model for general audio embedding/classifier"""

import tensorflow as tf
import tensorflow_hub
import numpy as np
import csv
import io
import pandas as pd


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [
        display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)
    ]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


import numpy as np
import tensorflow_hub

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier


def yamnet(
    url="https://tfhub.dev/google/yamnet/1",
    input_duration=60,
):
    return YamNET(url, input_duration)


class YamNET(BaseClassifier):
    """load TF model hub google Perch model, wrap in OpSo class

    Args:
        url to model path (default is Google-Bird-Vocalization-Classifier v3)

    Returns:
        object with .predict(), .embed() etc methods

    Methods:
        predict: get per-audio-clip per-class scores in dataframe format; includes WandB logging
        generate_embeddings: make embeddings for audio data (feature vectors from penultimate layer)
        generate_embeddings_and_logits: returns (embeddings, logits)
    """

    def __init__(self, url="https://tfhub.dev/google/yamnet/1", input_duration=60):
        """load TF model hub google Perch model, wrap in OpSo TensorFlowHubModel class

        Args:
            url to model path (default is Perch v3)
            input_duration: (sec) this amount of audio in internally windowed into
                0.96 sec clips with 0.48 sec overlap and batched for inference.
                This implicitly determines the batch size.

        Returns:
            object with .predict(), .embed() etc methods
        """

        # Load the model.
        self.network = tensorflow_hub.load(url)
        self.input_duration = input_duration
        self.preprocessor = AudioPreprocessor(
            sample_duration=input_duration, sample_rate=32000
        )
        # the dataloader returns a list of AudioSample objects, with .data as audio waveform samples
        self.inference_dataloader_cls = SafeAudioDataloader

        # load class list (based on example from https://tfhub.dev/google/yamnet/1)
        class_map_path = self.network.class_map_path().numpy()
        class_names = class_names_from_csv(
            tf.io.read_file(class_map_path).numpy().decode("utf-8")
        )
        self.classes = class_names

    def __call__(self, dataloader, **kwargs):
        """kwargs are passed to SafeAudioDataloader init (num_workers, batch_size, etc)

        returns logits, embeddings, logmelspec

        see https://tfhub.dev/google/yamnet/1 for details

        ## Inputs to tfhub model:
        - audio waveform in [-1,1] sampled at 16 kHz, any duration
        - internally, windows into batches:
        sliding windows of length 0.96 seconds and hop 0.48 seconds
        - since batching is internal, choice of input length determines batch size?
        - and we have to manually re-create the start and end times of each window/frame?
        - discards incomplete frame at end (it seems)
        """
        if dataloader.batch_size > 1:
            raise ValueError("batch size must be 1 for YamNET")

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        logmelspec = []
        start_times = []
        files = []
        for i, batch in enumerate(tqdm(dataloader)):
            waveform = batch[0].data.samples  # batch is always only one AudioSample
            # 1d input is batched into windows/frames internally by YamNET
            batch_logits, batch_embeddings, batch_logmelspec = self.network(waveform)
            logits.extend(batch_logits.numpy().tolist())
            embeddings.extend(batch_embeddings.numpy().tolist())
            logmelspec.extend(batch_logmelspec.numpy().tolist())

            # frames of returned scores start every 0.48 sec, and are 0.96 sec long
            # the batch start/end time are determined by self.input_duration
            batch_start_times = np.arange(
                i * self.input_duration, (i + 1) * self.input_duration, 0.48
            )
            # we might get one extra start time at the end
            if len(batch_start_times) == len(batch_logits) + 1:
                batch_start_times = batch_start_times[: len(batch_logits)]

            start_times.extend(batch_start_times)
            files.extend([batch[0].source] * len(batch_logits))

        # TODO: handle outputs by calculating start and end times of each frame
        return (
            logits,
            embeddings,
            logmelspec,
            start_times,
            files,
        )

    def generate_embeddings(self, samples, **kwargs):
        """Generate embeddings for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        Returns:
            list of embeddings, shape [N, 1024]
            Note: more return values than rows in input df, since YamNET internally
            batches and windows the audio into 0.96 sec clips with 0.48 sec overlap
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        _, embeddings, _, _, _ = self(dataloader)
        return embeddings

    def generate_embeddings_df(self, samples, **kwargs):
        """Generate embeddings for audio data

        Args:
            see self.generate_embeddings

        Returns:
            pd.DataFrame of embedding vectors, wwith (file, start_time, end_time) as index
            Note: returns more rows than inputs because of internal frame/windowing
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        _, embeddings, _, start_times, files = self(dataloader)
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [files, start_times, np.array(start_times) + 0.96],
                names=["file", "start_time", "end_time"],
            ),
            data=embeddings,
        )

    def generate_logits(self, samples, **kwargs):
        """Return (logits, embeddings) for audio data

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        Returns:
            list of 512 output scores, one per class (check self.classes for class names)
            (shape [N, 512])

            Note: more return values than rows in input df, since YamNET internally
            batches and windows the audio into 0.96 sec clips with 0.48 sec overlap
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        logits, _, _, _, _ = self(dataloader)
        return logits

    def generate_logmelspecs(self, samples, **kwargs):
        """Return 2d logmelspec arrays for audio data

        Args:
            samples: eithr of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        Returns:
            list of 2d logmelspec arrays (shape [n, m, 64])

            Note: one logmelspec per row in input df (or per file in list of files)
            n: number of inputs
            m: number of frames per input (input_length // .48 or one less)
        """
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        _, _, logmelspecs, _, _ = self(dataloader)
        return logmelspecs

    def predict(
        self,
        samples,
        **kwargs,
    ):
        """Generate predictions on a set of samples

        Return dataframe of model output scores for each sample.
        Optional activation layer for scores
        (softmax, sigmoid, softmax then logit, or None)

        Args:
            samples:
                the files to generate predictions for. Can be:
                - a dataframe with index containing audio paths, OR
                - a dataframe with multi-index (file, start_time, end_time), OR
                - a list (or np.ndarray) of audio file paths
            **kwargs: additional arguments to inference_dataloader_cls.__init__

        Note: batch size is always 1 since YamNET internally batches and windows.
            Use longer input_duration when initializing YamNet for larger batch size.

        Returns:
            dataframe of per-class scores with one row per input frame
            (0.96 sec long, 0.48 sec overlap between frames), since YamNET
            internally batches and windows the audio

            Note: returns more rows than inputs because of internal frame/windowing
        """
        kwargs.update({"batch_size": 1})
        # since YamNET internally batches and windows the audio, it makes sense to use partial
        # "remainder" mode even if we get less than sample_duration, eg 60 sec
        kwargs.update({"final_clip": "remainder"})
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        scores, _, _, start_times, files = self(dataloader)
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [files, start_times, np.array(start_times) + 0.96],
                names=["file", "start_time", "end_time"],
            ),
            data=scores,
            columns=self.classes,
        )
