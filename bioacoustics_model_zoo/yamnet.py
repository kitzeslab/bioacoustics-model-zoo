"""load the pre-trained YAMNet model for general audio embedding/classifier"""
import numpy as np
import csv
import io
import pandas as pd

from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [
        display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)
    ]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


class YAMNet(BaseClassifier):
    def __init__(self, url="https://tfhub.dev/google/yamnet/1", input_duration=60):
        """load TF model hub google Perch model, wrap in OpSo TensorFlowHubModel class

        Args:
            url to model path (default is YAMNet v1)
            input_duration: (sec) this amount of audio in internally windowed into
                0.96 sec clips with 0.48 sec overlap and batched for inference.
                This implicitly determines the batch size.

        Returns:
            object with .predict(), .generate_embeddings() etc methods

        Methods:
            predict (alias for generate_logits): get per-audio-clip per-class scores in dataframe format
            generate_embeddings: returns dataframe of embeddings (features from penultimate layer)
            generate_logmelspecs: returns list of 2d log-valued mel spectrogram arrays
            generate_embeddings_and_logits: returns 2 dfs (embeddings, logits)

        Example:
        ```
        import torch
        m=torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'YAMNet')
        m.predict(['test.wav']) # returns dataframe of per-class scores
        m.generate_embeddings(['test.wav']) # returns dataframe of embeddings
        ```
        """

        # only require tensorflow and tensorflow_hub if/when this class is used
        try:
            import tensorflow as tf
            import tensorflow_hub
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YAMNet requires tensorflow and tensorflow_hub packages to be installed. "
                "Install in your python environment with `pip install tensorflow tensorflow_hub`"
            ) from exc

        # Load the model.
        self.network = tensorflow_hub.load(url)
        self.input_duration = input_duration
        self.preprocessor = AudioPreprocessor(
            sample_duration=input_duration, sample_rate=16000
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
        """Run inference on a dataloader

        returns logits, embeddings, logmelspec, start_times, files

        see https://tfhub.dev/google/yamnet/1 for details

        ## Notes on inputs to this tfhub model:
        - audio waveform in [-1,1] sampled at 16 kHz, any duration
        - internally, windows into batches:
        sliding windows of length 0.96 seconds and hop 0.48 seconds
        - since batching is internal, choice of input length determines batch size?
        - and we have to manually re-create the start and end times of each window/frame?
        - discards incomplete frame at end (it seems)
        """
        if dataloader.batch_size > 1:
            raise ValueError("batch size must be 1 for YAMNet")

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        logmelspec = []
        start_times = []
        files = []
        for i, batch in enumerate(tqdm(dataloader)):
            waveform = batch[0].data.samples  # batch is always only one AudioSample
            # 1d input is batched into windows/frames internally by YAMNet
            batch_logits, batch_embeddings, batch_logmelspec = self.network(waveform)
            logits.extend(batch_logits.numpy().tolist())
            embeddings.extend(batch_embeddings.numpy().tolist())
            logmelspec.extend(batch_logmelspec.numpy().tolist())

            # frames of returned scores start every 0.48 sec, and are 0.96 sec long
            # the batch start/end time are determined by self.input_duration
            batch_start_times = np.arange(
                i * self.input_duration, (i + 1) * self.input_duration, 0.48
            )
            # we might get extra start time at the end
            batch_start_times = batch_start_times[: len(batch_logits)]
            start_times.extend(batch_start_times)

            # same file repeated for all frames in batch
            # AudioSample.source is the file path
            files.extend([batch[0].source] * len(batch_logits))

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
            pd.DataFrame of embedding vectors, wwith (file, start_time, end_time) as index
            Note: more return values than inputs, since YAMNet internally
            batches and windows the audio into 0.96 sec clips with 0.48 sec overlap

        """
        kwargs.update({"batch_size": 1})
        # since YAMNet internally batches and windows the audio, it makes sense to use partial
        # "remainder" mode even if we get less than sample_duration, eg 60 sec
        kwargs.update({"final_clip": "remainder"})
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        _, embeddings, _, start_times, files = self(dataloader)
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [files, start_times, np.array(start_times) + 0.96],
                names=["file", "start_time", "end_time"],
            ),
            data=embeddings,
        )

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
        kwargs.update({"batch_size": 1})
        # since YAMNet internally batches and windows the audio, it makes sense to use partial
        # "remainder" mode even if we get less than sample_duration, eg 60 sec
        kwargs.update({"final_clip": "remainder"})
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        _, _, logmelspecs, _, _ = self(dataloader)
        return logmelspecs

    def generate_logits(
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

        Note: batch size is always 1 since YAMNet internally batches and windows.
            Use longer input_duration when initializing YAMNet for larger batch size.

        Returns:
            dataframe of per-class scores with one row per input frame
            (0.96 sec long, 0.48 sec overlap between frames), since YAMNet
            internally batches and windows the audio

            Note: returns more rows than inputs because of internal frame/windowing
        """
        kwargs.update({"batch_size": 1})
        # since YAMNet internally batches and windows the audio, it makes sense to use partial
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

    predict = generate_logits  # alias

    def generate_embeddings_and_logits(self, samples, **kwargs):
        """returns 2 dfs (embeddings, logits) - see generate_logits and generate_embeddings

        Only runs inference once, so faster than calling both methods separately.

        Args:
            samples: any of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            **kwargs: any arguments to SafeAudioDataloader

        Returns:
            (embeddings, logits) dataframes
        """
        # borrows from generate_logits and generate_embeddings code
        # avoids re-running inference if both outputs are desired

        kwargs.update({"batch_size": 1})
        # since YAMNet internally batches and windows the audio, it makes sense to use partial
        # "remainder" mode even if we get less than sample_duration, eg 60 sec
        kwargs.update({"final_clip": "remainder"})
        dataloader = self.inference_dataloader_cls(samples, self.preprocessor, **kwargs)
        scores, embeddings, _, start_times, files = self(dataloader)

        embedding_df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [files, start_times, np.array(start_times) + 0.96],
                names=["file", "start_time", "end_time"],
            ),
            data=embeddings,
        )

        score_df = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [files, start_times, np.array(start_times) + 0.96],
                names=["file", "start_time", "end_time"],
            ),
            data=scores,
            columns=self.classes,
        )

        return embedding_df, score_df
