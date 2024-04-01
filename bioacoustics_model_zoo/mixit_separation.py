# code from https://github.com/google-research/sound-separation/blob/master/models/tools/process_wav.py
import os
from opensoundscape.audio import Audio
from pathlib import Path
import numpy as np
import warnings


class SeparationModel(object):
    """Tensorflow audio separation model."""

    def __init__(
        self,
        checkpoint,
        metagraph_path=None,
        input_tensor_name="input_audio/receiver_audio:0",
        output_tensor_name="denoised_waveforms:0",
    ):
        """Initializes the separation model from checkpoint

        note: to use this class with torch.hub.load(), you must provide the
        checkpoint='/path/to/downloaded/checkpoint'. To download the MixIt checkpoint,
        run `gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .`
        or see instructions on the [MixIt Github repo](https://github.com/google-research/sound-separation/blob/master/models/bird_mixit/README.md)

        Args:
            checkpoint: location of the checkpoint to load
                eg /path/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090
                Note that ".data-00000-of-00001" or ".index" is not included
            metagraph_path: location of the metagraph to load
                if None, looks in same folder as checkpoint for "inference.meta"
            input_tensor_name: name of the input tensor (typically use default)
            output_tensor_name: name of the output tensor (typically use default)

        Methods:
            separate_waveform: separates a mixture waveform np array into sources
                (returns np array [batch, num_sources, num_samples]])
            separate_audio: separates an opensoundscape.Audio object into sources
                (returns list of Audio)

        Example:

        First, download the checkpoint and metagraph from the MixIt Github repo:
        install gsutil and run the following command in your terminal:
        `gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .`

        Then, use the model in python:
        ```
        import torch
        # provide the local path to the checkpoint when creating the object
        model = torch.hub.load(
            'kitzeslab/bioacoustics-model-zoo',
            'SeparationModel',
            checkpoint='/path/to/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090',
            trust_repo=True
        ) #creates 4 channels; use output_sources8 to separate into 8 channels

        # separate opensoundscape Audio object into 4 channels:
        # note that it seems to work best on 5 second segments
        a=Audio.from_file('audio.mp3',sample_rate=22050).trim(0,5)
        separated = model.separate_audio(a)

        # save audio files for each separated channel:
        # saves audio files with extensions like _stem0.wav, _stem1.wav, etc
        model.load_separate_write('./temp.wav')
        ```

        """
        # only require tensorflow if/when this class is used
        try:
            import tensorflow.compat.v1 as tf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SeparationModel requires tensorflow package to be installed. "
                "Install in your python environment with `pip install tensorflow`"
            ) from exc
        # TODO allow download of checkpoint from a public url without gsutil

        # make sure checkpoint doesn't include .data-00000-of-00001 or .index
        suffix = Path(checkpoint).suffix
        if "data" in suffix or "index" in suffix:
            warnings.warn(
                "checkpoint should not include .data-... or .index. Removing suffix."
            )
            checkpoint = checkpoint.replace(suffix, "")

        if metagraph_path is None:
            metagraph_path = Path(checkpoint).parent / "inference.meta"
            assert (
                metagraph_path.exists()
            ), f"'inference.meta' not found in same folder as checkpoint ({Path(checkpoint).parent}) and not specified"

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            tf.logging.info("Importing meta graph: %s", metagraph_path)
            new_saver = tf.train.import_meta_graph(metagraph_path)
            print("Restoring model from checkpoint: ", checkpoint)
            new_saver.restore(self.sess, checkpoint)
        self.input_placeholder = self.graph.get_tensor_by_name(input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)
        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name

    def separate_waveform(self, mixture_waveform):
        """Separates a mixture waveform into sources.

        Args:
          mixture_waveform: numpy.ndarray of shape (batch, num_mics, num_samples),
            or (num_mics, num_samples,) or (num_samples,). Currently only works
            when batch=1 for three dimensional inputs.

        Returns:
          numpy.ndarray of (num_sources, num_samples) of source estimates.
        """
        num_input_dims = np.ndim(mixture_waveform)
        if num_input_dims == 1:
            mixture_waveform_input = mixture_waveform[np.newaxis, np.newaxis, :]
        elif num_input_dims == 2:
            mixture_waveform_input = mixture_waveform[np.newaxis, :]
        elif num_input_dims == 3:
            assert np.shape(mixture_waveform)[0] == 1
            mixture_waveform_input = mixture_waveform
        else:
            raise ValueError(
                "Unsupported number of mixture waveform input "
                f"dimensions {num_input_dims}."
            )

        separated_waveforms = self.sess.run(
            self.output_tensor,
            feed_dict={self.input_placeholder: mixture_waveform_input},
        )[0]
        return separated_waveforms

    def separate_audio(self, audio):
        """Separate audio object into 4 components

        Args:
            audio: opensoundscape.Audio object

        Returns:
            list of opensoundscape.Audio objects
        """
        # resample to 22050 and make the 3-d np array that the TF model expects
        waveform = audio.resample(22050).samples[np.newaxis, np.newaxis, :]
        separated = self.separate_waveform(waveform)  # [num_sources, num_samples]
        return [Audio(s, sample_rate=audio.sample_rate) for s in separated]

    def load_separate_write(
        self,
        input_path,
        output_path=None,
    ):
        """loads audio file, separates into sources, and writes new files

        Audio is resampled to 22050 Hz. Can be any length.

        Args:
            input_path (str): path to input audio file
            output_path (str): path to output files:
                - extension can be any supported by SoundFile ('wav','mp3', etc)
                - channel will be appended to filename. Eg: input.wav -> input_stem0.wav
                - if None, output_path = input_path [default: None] (does not overwrite because
                    extensions are added to each separated audio stem)
        """
        if output_path is None:
            output_path = input_path  # save in same place as input file
        output_path = Path(output_path)
        # load audio and resample to 22050 Hz
        audio = Audio.from_file(input_path, sample_rate=22050)
        # run separation model
        separated_audio = self.separate_audio(audio)
        # save each source as a separate file
        for i, audio in enumerate(separated_audio):
            save_path = output_path.with_stem(output_path.stem + f"_stem{i}")
            audio.save(save_path)
