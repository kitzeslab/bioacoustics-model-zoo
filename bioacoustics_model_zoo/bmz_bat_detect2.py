import numpy as np
import pandas as pd
import torch
from opensoundscape.ml.cnn import SpectrogramClassifier
from opensoundscape.preprocess.actions import BaseAction
from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor

from bioacoustics_model_zoo.utils import register_bmz_model


class BatDetect2PreprocessorAction(BaseAction):
    """action taking opso.Audio and returning a spectrogram tensor for batdetect2"""

    def __init__(self, batdetect_preprocessor):
        super().__init__()
        # how to make this thread safe? batdetect2 preprocessor is not thread safe, so we need to make a new one for each worker
        self.batdetect_preprocessor = batdetect_preprocessor

    def __call__(self, sample):
        # first extract the samples from Audio object, cast to torch tensor, and add channel dim
        audio_tensor = torch.as_tensor(sample.data.samples).unsqueeze(0)
        # run the provided preprocessor object to generate spectrogram
        sample.data = self.batdetect_preprocessor(audio_tensor)


class BatDetect2Pre(AudioAugmentationPreprocessor):
    def __init__(
        self,
        batdetect_preprocessor,
        sample_duration,
        sample_rate,
        extend_short_clips=False,
        overlay_samples=None,
    ):
        super().__init__(
            sample_duration, sample_rate, extend_short_clips, overlay_samples
        )
        # don't trim or extend clips, just the duration provided
        self.remove_action("trim_audio")

        preprocessor_action = BatDetect2PreprocessorAction(batdetect_preprocessor)

        self.insert_action("batdetect_preprocessor", preprocessor_action)


@register_bmz_model
class BatDetect2(SpectrogramClassifier):
    def __init__(self, checkpoint_path=None, sample_duration=10):
        try:
            import batdetect2
        except ImportError:
            raise ImportError(
                "batdetect2 is required to use this model and was not found in the environment. Please install it with `pip install --pre --upgrade batdetect2`"
            )
        bd = batdetect2.BatDetect2API.from_checkpoint(checkpoint_path)
        super().__init__(
            architecture=bd.model.detector,
            classes=bd.targets.class_names,
            sample_duration=10,
            sample_rate=bd.audio_config.samplerate,
        )
        # no MPS support for 'take' operator
        if self.device.type == "mps":
            self.device = torch.device("cpu")
        self.preprocessor = BatDetect2Pre(
            batdetect_preprocessor=bd.preprocessor,
            sample_duration=sample_duration,
            sample_rate=bd.audio_config.samplerate,
        )
        self.bd = bd
        # hard coding this could break custom checkpoints -> override after init if needed
        self.embedding_size = 32
        self.sort_returns_by_time = True

    def batch_forward(
        self, batch, targets=["detection_probs", "class_probs", "features"]
    ):
        """forward pass through the model, returning a dict of outputs

        batching not supported apparently (haven't tried though)
        """
        outputs = []
        for sample in batch:
            spatial_outputs = self.bd.model.detector(sample.data.unsqueeze(0))
            # has attributes .detection_probs, .class_probs, .features
            # post-process to detections here?
            outputs.append(spatial_outputs)
        returns = {
            target: torch.stack([getattr(o, target) for o in outputs])
            for target in targets
            if targets != -1
        }
        if -1 in targets:
            returns["class_probs"] = outputs.class_probs

        return returns

    def detect(self, samples, detection_threshold=0.5, **kwargs):
        """detect sound events and return per-class prediction scores for each event

        Args:
            samples: file path, list of paths, or dataframe defining audio clip file:start:end times
            detection_threshold: float, threshold for detection confidence score
            **kwargs: additional keyword arguments passed to self.predict_dataloader, e.g. batch_size, num_workers

        Returns:
            pd.DataFrame with columns:
                - file: str, path to audio file
                - start_time: float, start time of detection in seconds
                - end_time: float, end time of detection in seconds
                - low_frequency: float, low frequency of detection in Hz
                - high_frequency: float, high frequency of detection in Hz
                - detection_score: float, confidence score of the detection
                - class_1, class_2, ..., class_n: float, confidence score for each class for each detection
        """
        dl = self.predict_dataloader(samples=samples, **kwargs)
        self.bd.model.to(self.device)
        all_scores = []
        for batch in dl:
            batch_dets = []
            for sample in batch:
                outs = self.bd.process_spectrogram(
                    sample.data.to(self.device),
                    detection_threshold=detection_threshold,
                )
                dets = [self._det_to_class_scores(det, sample) for det in outs]
                if self.sort_returns_by_time:
                    dets.sort(key=lambda x: x["start_time"])
                batch_dets.extend(dets)

            if self.sort_returns_by_time:
                batch_dets.sort(key=lambda x: x["start_time"])
            all_scores.extend(batch_dets)
        if len(all_scores) == 0:
            return self._empty_label_df()

        return pd.DataFrame(all_scores)

    def label(self, samples, detection_threshold=0.5, **kwargs):
        """detect sound events, and return a dataframe with a single selected class and confidence score for each detection

        Args:
            samples: file path, list of paths, or dataframe defining audio clip file:start:end times
            detection_threshold: float, threshold for detection confidence score
            **kwargs: additional keyword arguments passed to self.predict_dataloader, e.g. batch_size, num_workers

        Returns:
            pd.DataFrame with columns:
                - file: str, path to audio file
                - start_time: float, start time of detection in seconds
                - end_time: float, end time of detection in seconds
                - low_frequency: float, low frequency of detection in Hz
                - high_frequency: float, high frequency of detection in Hz
                - class: str, selected class for each detection
                - score: float, confidence score for each detection
                - detection_score: float, confidence score of the detection
        """
        dets = self.detect(samples, detection_threshold=detection_threshold, **kwargs)
        dets["class"] = dets[self.classes].idxmax(axis=1)
        dets["score"] = dets[self.classes].max(axis=1)
        return dets[
            [
                "file",
                "start_time",
                "end_time",
                "low_frequency",
                "high_frequency",
                "class",
                "score",
                "detection_score",
            ]
        ]

    def features(self, samples, detection_threshold=0.5, **kwargs):
        """detect sound events, and return a dataframe of features for each detection

        Args:
            samples: file path, list of paths, or dataframe defining audio clip file:start:end times
            detection_threshold: float, threshold for detection confidence score
            **kwargs: additional keyword arguments passed to self.predict_dataloader, e.g. batch_size, num_workers

        Returns:
            pd.DataFrame with columns:
                - file: str, path to audio file
                - start_time: float, start time of detection in seconds
                - end_time: float, end time of detection in seconds
                - low_frequency: float, low frequency of detection in Hz
                - high_frequency: float, high frequency of detection in Hz
                - detection_score: float, confidence score of detection
                - 0, 1, ..., embedding_size-1: float, features for each detection
        """
        dl = self.predict_dataloader(samples=samples, **kwargs)
        self.bd.model.to(self.device)
        all_feats = []
        for batch in dl:
            batch_feats = []
            for sample in batch:
                outs = self.bd.process_spectrogram(
                    sample.data.to(self.device),
                    detection_threshold=detection_threshold,
                )
                batch_feats.extend([self._det_to_features(det, sample) for det in outs])
            all_feats.extend(batch_feats)
        if len(all_feats) == 0:  # empty dataframe with correct columns
            return self._empty_features_df()
        return pd.DataFrame(all_feats)

    def _empty_label_df(self):
        return pd.DataFrame(
            columns=[
                "file",
                "start_time",
                "end_time",
                "low_frequency",
                "high_frequency",
                "detection_score",
            ]
            + self.bd.targets.class_names
        )

    def _empty_features_df(self):
        return pd.DataFrame(
            columns=[
                "file",
                "start_time",
                "end_time",
                "low_frequency",
                "high_frequency",
                "detection_score",
            ]
            + list(range(self.embedding_size))
        )

    def _det_to_class_scores(self, detection, sample):
        det = {
            "file": sample.source,
            "start_time": sample.start_time + detection.geometry.coordinates[0],
            "end_time": sample.start_time + detection.geometry.coordinates[2],
            "low_frequency": detection.geometry.coordinates[1],
            "high_frequency": detection.geometry.coordinates[3],
            "detection_score": detection.detection_score,
        }
        class_scores = {
            self.classes[i]: score for i, score in enumerate(detection.class_scores)
        }
        return {**det, **class_scores}

    def _det_to_features(self, detection, sample):
        det = {
            "file": sample.source,
            "start_time": sample.start_time + detection.geometry.coordinates[0],
            "end_time": sample.start_time + detection.geometry.coordinates[2],
            "low_frequency": detection.geometry.coordinates[1],
            "high_frequency": detection.geometry.coordinates[3],
            "detection_score": detection.detection_score,
        }
        features = {i: f for i, f in enumerate(detection.features)}
        return {**det, **features}

    def predict(self, samples, agg="max", **kwargs):
        """Per-class confidence scores on fixd-duration audio clips (not bounding boxes)

        Args:
            samples: file path, list of paths, or dataframe defining audio clip file:start:end times
            agg: 'max' or 'mean', how to aggregate per-detection class scores into a single score
                for each fixed-duration clip

            **kwargs: additional keyword arguments passed to self.predict_dataloader, e.g. batch_size, num_workers
        """
        dl = self.predict_dataloader(samples=samples, **kwargs)
        self.bd.model.to(self.device)
        all_scores = []
        for batch in dl:
            batch_scores = []
            for sample in batch:
                outs = self.bd.process_spectrogram(
                    sample.data.to(self.device),
                    detection_threshold=0,
                )
                all_class_scores = np.stack([det.class_scores for det in outs])
                if agg == "max":
                    class_scores = all_class_scores.max(axis=0)
                elif agg == "mean":
                    class_scores = all_class_scores.mean(axis=0)
                else:
                    raise ValueError("Agg must be 'max' or 'mean'")
                batch_scores.append(class_scores)
            all_scores.extend(batch_scores)
        return pd.DataFrame(
            index=dl.dataset.dataset.label_df.index,
            columns=self.classes,
            data=np.stack(all_scores),
        )
