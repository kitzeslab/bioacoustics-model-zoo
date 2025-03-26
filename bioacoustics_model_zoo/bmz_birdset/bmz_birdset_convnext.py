from transformers import ConvNextForImageClassification
from opensoundscape import SpectrogramClassifier
from opensoundscape.preprocess.preprocessors import register_preprocessor_cls
from bioacoustics_model_zoo.bmz_birdset.birdset_preprocessing import BirdsetPreprocessor
from bioacoustics_model_zoo.utils import register_bmz_model


@register_preprocessor_cls
class ConvNextBirdsetPreprocessor(BirdsetPreprocessor):
    """hard-code preprocessing parameters used for Birdset ConvNext XCL model"""

    def __init__(self, sample_duration=5, sample_rate=32000, overlay_df=None):
        super().__init__(
            sample_duration=sample_duration,
            sample_rate=sample_rate,
            overlay_df=overlay_df,
        )
        self.pipeline.to_spec.set(n_fft=1024, hop_length=320, power=2.0)
        self.pipeline.to_mel.set(n_mels=128, n_stft=513)
        self.pipeline.power_to_db.set(top_db=80)
        self.pipeline.rescale.set(input_mean=-4.268, input_std=4.569)


class ConvNextForImageClassificationLogits(ConvNextForImageClassification):
    """simplfy forward call to just return logits

    parent class forward call returns object containing logits, loss, intermediate outputs
    """

    def forward(self, pixel_values=None, labels=None, **kwargs):
        # do not pass labels, so that loss is not computed
        # we compute the loss elsewhere if required
        # TODO: revisit - might be faster to allow forward to compute loss?
        return super().forward(pixel_values, labels=None, **kwargs).logits


@register_bmz_model
class BirdSetConvNeXT(SpectrogramClassifier):
    def __init__(self):
        """BirdSet ConvNeXT global bird species foundation model

        > Rauch, Lukas, et al. "Birdset: A multi-task benchmark for classification in avian bioacoustics." arXiv e-prints (2024): arXiv-2403.

        BirdSet GitHub: https://github.com/DBD-research-group/BirdSet

        by default loads weights of model trained on Xeno Canto full (XCL Dataset)

        to prepare an environment run:
        ```
        conda create -n birdset python=3.10
        conda activate birdset
        pip install opensoundscape transformers torch torchvision torchaudio
        ```

        Implements standard api: .train(), .predict(), .embed(), .generate_cams(),
        .generate_samples()
        """

        model = ConvNextForImageClassificationLogits.from_pretrained(
            "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            cache_dir=".",
            ignore_mismatched_sizes=True,
        )
        # note that this class list seems to use the Birdnet 2022 taxonomy
        # use name_conversions github repo to convert between these codes and common/scientific names
        classes = [model.config.id2label[i] for i in range(model.num_labels)]

        super().__init__(model, classes=classes, sample_duration=5)

        self.preprocessor = ConvNextBirdsetPreprocessor()
        self.network.to(self.device)

        # define default classifier, embedding, and gradcam layers
        self.network.classifier_layer = "classifier"
        self.network.embedding_layer = "convnext.layernorm"
        self.network.cam_layer = "convnext.encoder.stages.2.layers.26"
