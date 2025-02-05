# install:
# pip install -e git+https://github.com/DBD-research-group/BirdSet.git#egg=birdset
import datasets
from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape.preprocess.actions import BaseAction
from opensoundscape import SpectrogramClassifier

import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

import os
import hydra
from omegaconf import DictConfig, OmegaConf

# from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
import librosa
from transformers import ConvNextForImageClassification
import torch
from pathlib import Path


# note that I copied the entire "configs" folder from the birdset repo
# to the current working directory
try:
    hydra.initialize(config_path="configs")  # relative to current working dir
except:
    print("hydra already initialized?")
os.environ["PROJECT_ROOT"] = "root"


class BirdSetAction(BaseAction):
    def __init__(self, cfg):
        super().__init__()

        # get the transform config
        transform_cfg = cfg.datamodule.transforms

        # to allow del
        OmegaConf.set_struct(transform_cfg, False)

        # delete background noise since an error could occur when the path has no files (even if it isn't used)
        # TODO: revisit
        del transform_cfg.waveform_augmentations["background_noise"]
        self.birdset_transform = hydra.utils.instantiate(transform_cfg)

    def __call__(self, sample):
        batch = {
            "audio": [{"array": sample.data.samples}],
            "labels": torch.tensor(sample.labels.values).unsqueeze(1),
        }
        samples, labels = self.birdset_transform.transform_values(batch)
        sample.data = samples[0]
        sample.labels = pd.Series(labels[0].numpy(), index=sample.labels.index)


class BirdSetPreprocessor(AudioAugmentationPreprocessor):
    def __init__(self, cfg=None):

        if cfg is None:
            # load default config
            cfg = hydra.compose(
                config_name="train",
                overrides=["experiment=birdset_neurips24/XCL/convnext"],
            )

        ##  create the preprocessor ##
        # TODO: access the config for these parameters rather than hard-coding
        super().__init__(sample_duration=5, sample_rate=32000)
        self.insert_action("birdset_transform", BirdSetAction(cfg))


class ConvNextForImageClassificationLogits(ConvNextForImageClassification):
    """simplfy forward call to just return logits

    parent class forward call returns object containing logits, loss, intermediate outputs
    """

    def forward(self, pixel_values=None, labels=None, **kwargs):
        # do not pass labels, so that loss is not computed
        # we compute the loss elsewhere if required
        return super().forward(pixel_values, labels=None, **kwargs).logits


class BirdSet(SpectrogramClassifier):
    def __init__(self):
        # TODO config for inference without augmentation?
        model = ConvNextForImageClassificationLogits.from_pretrained(
            "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            cache_dir=".",
            ignore_mismatched_sizes=True,
        )
        dataset_meta = datasets.load_dataset_builder(
            "dbd-research-group/BirdSet", "XCL"
        )
        classes = dataset_meta.info.features["ebird_code"]
        class_list = np.array(classes.names)

        super().__init__(model, classes=class_list, sample_duration=5)

        self.preprocessor = BirdSetPreprocessor()
        self.network.to(self.device)
