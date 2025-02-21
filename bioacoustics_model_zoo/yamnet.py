"""load the pre-trained YAMNet model for general audio embedding/classifier"""

import numpy as np
import csv
import io
import pandas as pd

import opensoundscape
from opensoundscape.preprocess.preprocessors import AudioPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from tqdm.autonotebook import tqdm
from opensoundscape.ml.cnn import BaseClassifier
from opensoundscape import Audio, Action

from bioacoustics_model_zoo.utils import register_bmz_model


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [
        display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)
    ]
    class_names = class_names[1:]  # Skip CSV header
    return class_names


class YAMNetDataloader(SafeAudioDataloader):
    """wraps SafeAudioDataloader to return sample arrays"""

    def __init__(self, *args, **kwargs):
        kwargs.update({"batch_size": 1})
        # since YAMNet internally batches and windows the audio, it makes sense to use partial
        # "remainder" mode even if we get less than sample_duration, eg 60 sec
        kwargs.update({"final_clip": "remainder"})
        kwargs.update({"collate_fn": opensoundscape.utils.identity})
        super().__init__(*args, **kwargs)


@register_bmz_model
class YAMNet(BaseClassifier):
    def __init__(self, url="https://tfhub.dev/google/yamnet/1", input_duration=60):
        """load YAMNet Audio CNN from tensorflow hub

        Args:
            url to model path (default is YAMNet v1)
            input_duration: (sec) this amount of audio in internally windowed into
                0.96 sec clips with 0.48 sec overlap and batched for inference.
                This implicitly determines the batch size.

        Returns:
            object with .predict(), .generate_embeddings() etc methods

        Methods:
            predict (alias for generate_logits): get per-audio-clip per-class scores in dataframe format
            embed: returns dataframe of embeddings (features from penultimate layer); optionally also return predictions
            generate_logmelspecs: returns np.array of 2d log-valued mel spectrogram arrays

        Example:
        ```
        import bioacoustics_model_zoo as bmz
        m=bmz.YAMNet()
        m.predict(['test.wav']) # returns dataframe of per-class scores
        m.embed(['test.wav']) # returns dataframe of embeddings
        m.generate_logmelspecs(['test.wav']) # returns np.array of logmelspecs
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
        # extend short samples to input_duration by padding end with zeros (silence)
        self.preprocessor.insert_action(
            action_index="extend",
            action=Action(
                Audio.extend_to, is_augmentation=False, duration=input_duration
            ),
        )
        # the dataloader returns a list of AudioSample objects, with .data as audio waveform samples
        self.inference_dataloader_cls = YAMNetDataloader  # SafeAudioDataloader

        # load class list (based on example from https://tfhub.dev/google/yamnet/1)
        class_map_path = self.network.class_map_path().numpy()
        class_names = class_names_from_csv(
            tf.io.read_file(class_map_path).numpy().decode("utf-8")
        )
        self.classes = class_names

        self.device = opensoundscape.ml.cnn._gpu_if_available()

    def __display__(self):
        return "YAMNnet model loaded from tfhub"

    def __repr__(self):
        return "YAMNnet model loaded from tfhub"

    def __call__(
        self,
        dataloader,
        wandb_session=None,
        progress_bar=True,
    ):
        """Run inference on a dataloader created with self.predict_dataloader()

        see https://tfhub.dev/google/yamnet/1 for details

        ## Notes on inputs to this tfhub model:
        - audio waveform in [-1,1] sampled at 16 kHz, any duration
        - internally, windows into batches:
        sliding windows of length 0.96 seconds and hop 0.48 seconds
        - since batching is internal, choice of input length determines batch size?
        - and we have to manually re-create the start and end times of each window/frame?
        - discards incomplete frame at end (it seems)

        Args:
            dataloader: a dataloader created with self.predict_dataloader()

        Returns:
            if return_embeddings=True, returns (logits, embeddings)
            otherwise, returns logits
        """
        if dataloader.batch_size > 1:
            raise ValueError("batch size must be 1 for YAMNet")

        # iterate batches, running inference on each
        logits = []
        embeddings = []
        logmelspecs = []
        start_times = []
        files = []
        for i, batch in enumerate(tqdm(dataloader, disable=not progress_bar)):
            # 1d input is batched into windows/frames internally by YAMNet
            sample_array = batch[0].data.samples
            # discard logmelspec return value to avoid large memory usage
            batch_logits, batch_embeddings, batch_specs = self.network(sample_array)
            logits.extend(batch_logits.numpy().tolist())
            embeddings.extend(batch_embeddings.numpy().tolist())
            logmelspecs.extend(batch_specs.numpy().tolist())

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

            if wandb_session is not None:
                wandb_session.log(
                    {
                        "progress": i / len(dataloader),
                        "completed_batches": i,
                        "total_batches": len(dataloader),
                    }
                )

        return (
            np.array(logits),
            np.array(embeddings),
            np.array(logmelspecs),
            start_times,
            files,
        )

    def predict(self, samples, progress_bar=True, wandb_session=None, **kwargs):
        """
        Generate per-class scores for audio files/clips

        Args:
            samples: either of the following:
                - list of file paths
                - Dataframe with file as index
                - Dataframe with file, start_time, end_time of clips as index
            progress_bar: bool, if True, shows a progress bar with tqdm [default: True]
            return_dfs: bool, if True, returns scores as pd.DataFrame with multi-index like
                .predict(). if False, returns np.array of scores [default: True].
            kwargs are passed to self.predict_dataloader()

        Returns:
            pd.DataFrame of per-class scores if return_dfs=True, or np.array if return_dfs=False
            - scores are logit scores with no activation layer applied
        """
        # create dataloader to generate batches of AudioSamples
        dataloader = self.predict_dataloader(samples, **kwargs)

        # run inference; discard embeddings and logmelspecs
        preds, _, _, start_times, files = self(
            dataloader=dataloader,
            progress_bar=progress_bar,
            wandb_session=wandb_session,
        )

        # put predictions in a DataFrame with multi-index
        preds = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [files, start_times, np.array(start_times) + 0.96],
                names=["file", "start_time", "end_time"],
            ),
            data=preds,
        )

        return preds

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

        # run inference
        preds, embeddings, _, start_times, files = self(
            dataloader=dataloader,
            progress_bar=progress_bar,
        )

        if return_dfs:
            # put embeddings in DataFrame with multi-index like .predict()
            embeddings = pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [files, start_times, np.array(start_times) + 0.96],
                    names=["file", "start_time", "end_time"],
                ),
                data=embeddings,
            )

        if return_preds:
            if return_dfs:
                # put predictions in a DataFrame with same index as embeddings
                preds = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays(
                        [files, start_times, np.array(start_times) + 0.96],
                        names=["file", "start_time", "end_time"],
                    ),
                    data=preds,
                )
            return embeddings, preds
        else:
            return embeddings

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
        dataloader = self.predict_dataloader(samples, **kwargs)
        _, _, logmelspecs, _, _ = self(dataloader)
        return logmelspecs
