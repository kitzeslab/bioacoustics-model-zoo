from transformers import EfficientNetForImageClassification
from opensoundscape import SpectrogramClassifier
from opensoundscape.preprocess.preprocessors import register_preprocessor_cls
from bioacoustics_model_zoo.bmz_birdset.birdset_preprocessing import BirdsetPreprocessor
from bioacoustics_model_zoo.utils import register_bmz_model
from bioacoustics_model_zoo.cache import get_model_cache_dir


@register_preprocessor_cls
class EfficientnetBirdsetPreprocessor(BirdsetPreprocessor):
    """hard-code preprocessing parameters used for Birdset Efficientnet B0 XCL model"""

    def __init__(self, sample_duration=5, sample_rate=32000, overlay_df=None):
        super().__init__(
            sample_duration=sample_duration,
            sample_rate=sample_rate,
            overlay_df=overlay_df,
        )
        self.pipeline.to_spec.set(n_fft=2048, hop_length=256, power=2.0)
        self.pipeline.to_mel.set(n_mels=256, n_stft=1025, sample_rate=32000)
        self.pipeline.power_to_db.set(top_db=80)
        self.pipeline.rescale.set(input_mean=-4.268, input_std=4.569)


class EfficientNetLogits(EfficientNetForImageClassification):
    """simplfy forward call to just return logits

    parent class forward call returns object containing logits, loss, intermediate outputs
    """

    def forward(self, pixel_values=None, labels=None, **kwargs):
        # do not pass labels, so that loss is not computed
        # we compute the loss elsewhere if required
        return super().forward(pixel_values, labels=None, **kwargs).logits


@register_bmz_model
class BirdSetEfficientNetB1(SpectrogramClassifier):
    """BirdSet EfficientNetB1 global bird species foundation model

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

    Example: predict and embed:
    ```
    import bioacoustics_model_zoo as bmz
    m=bmz.BirdSetEfficientNetB1()
    m.predict(['test.wav'],batch_size=64) # returns dataframe of per-class scores
    m.embed(['test.wav']) # returns dataframe of embeddings
    ```

    Example: train on different set of classes (see OpenSoundscape tutorials for details on training)
    ```
    import bioacoustics_model_zoo as bmz
    import pandas as pd

    # load pre-trained network and change output classes
    m=bmz.BirdSetEfficientNetB1()
    m.change_classes(['crazy_zebra_grunt','screaming_penguin'])

    # optionally, freeze feature extractor (only train final layer)
    m.freeze_feature_extractor()

    # load one-hot labels and train (index: (file,start_time,end_time))
    train_df = pd.read_csv('train_labels.csv',index_col=[0,1,2])
    val_df = pd.read_csv('val_labels.csv',index_col=[0,1,2])
    m.train(train_df, val_df,batch_size=128, num_workers=8)
    ```
    """

    def __init__(self, cache_dir=None):

        # Get cache directory for BirdSet models
        hf_cache_dir = str(get_model_cache_dir("birdset_efficientnetb1", cache_dir))

        model = EfficientNetLogits.from_pretrained(
            "DBD-research-group/EfficientNet-B1-BirdSet-XCL",
            cache_dir=hf_cache_dir,
            ignore_mismatched_sizes=True,
        )
        # note that this class list seems to use the Birdnet 2022 taxonomy
        # use name_conversions github repo to convert between these codes and common/scientific names
        classes = [model.config.id2label[i] for i in range(model.num_labels)]

        super().__init__(model, classes=classes, sample_duration=5)

        self.preprocessor = EfficientnetBirdsetPreprocessor()
        self.network.to(self.device)

        # define default classifier, embedding, and gradcam layers
        self.network.classifier_layer = "classifier"
        self.network.embedding_layer = "efficientnet.pooler"
        self.network.cam_layer = "efficientnet.encoder.blocks.1"
