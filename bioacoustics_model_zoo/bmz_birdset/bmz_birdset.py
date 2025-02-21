# install:
# pip install -e git+https://github.com/sammlapp/BirdSet.git
# pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo.git
# pip install datasets transformers opensoundscape==0.11.0

import datasets  # huggingface datasets with access to BirdSet repo
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

        # delete background noise addition since an error could occur when the path has no files (even if it isn't used)
        # TODO: revisit
        del transform_cfg.waveform_augmentations["background_noise"]
        self.birdset_transform = hydra.utils.instantiate(transform_cfg)

    def __call__(self, sample):
        # preprocessing fails if augmentation is on and there are no labels
        # give a helpful error message in this case
        if len(sample.labels) == 0 and self.birdset_transform.mode == "train":
            raise ValueError(
                "Sample has no labels. Please turn off augmentations or provide labels. "
                "BirdSet augmentation operations require labels. "
                "Augmentation is off by default during predict() but on by default during train(). "
                "Augmentation can be turned off by setting `bypass_augmentations=True` in the `preprocessor.forward()` "
                "or `CNN.generate_samples()` calls."
            )
        batch = {
            "audio": [{"array": sample.data.samples}],
            "labels": torch.tensor(sample.labels.values).unsqueeze(1),
        }

        labels_values_dict = self.birdset_transform(batch)
        sample.data = labels_values_dict["input_values"].squeeze(0)
        if len(labels_values_dict["labels"]) > 0:
            sample.labels = pd.Series(
                labels_values_dict["labels"][:, 0].numpy(),  # remove batch dim
                index=sample.labels.index,
            )
        else:
            pass  # leave sample.labels without modification


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

    def forward(self, sample, bypass_augmentations=False, **kwargs):
        """process audio sample with BirdSet preprocessing operations

        identical to AudioAugmentationPreprocessor.forward, but
        switch BirdSetDatamoduleWrapper.mode between "train" and "predict"
        based on bypass_augmentations

        Args:
            sample (Sample): The sample to be processed.
            bypass_augmentations (bool, optional): Whether to bypass augmentations. Defaults to False.
            **kwargs: see AudioAugmentationPreprocessor.forward

        Returns:
            Sample: The processed AudioSample object with .data and .labels attributes
        """
        # # switch BirdSetDatamoduleWrapper.mode between "train" and "predict" based on bypass_augmentations
        self.pipeline.birdset_transform.birdset_transform.mode = (
            "predict" if bypass_augmentations else "train"
        )
        return super().forward(
            sample, bypass_augmentations=bypass_augmentations, **kwargs
        )


class ConvNextForImageClassificationLogits(ConvNextForImageClassification):
    """simplfy forward call to just return logits

    parent class forward call returns object containing logits, loss, intermediate outputs
    """

    def forward(self, pixel_values=None, labels=None, **kwargs):
        # do not pass labels, so that loss is not computed
        # we compute the loss elsewhere if required
        return super().forward(pixel_values, labels=None, **kwargs).logits


class BirdSetConvNeXT(SpectrogramClassifier):
    def __init__(self):
        """BirdSet pretrained ConvNeXT model

        by default loads weights of model trained on Xeno Canto full (XCL Dataset)

        to prepare an environment run:
        ```
        conda create -n birdset python=3.10
        conda activate birdset
        pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo.git
        pip install opensoundscape git+https://github.com/sammlapp/BirdSet.git#egg=birdset
        #pip install datasets transformers
        ```
        """
        # for no augmentation: self.preprocessor.pipeline.birdset_transform.birdset_transform.mode = "predict"
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

        # define default classifier, embedding, and gradcam layers
        self.network.classifier_layer = "classifier"
        self.network.embedding_layer = "convnext.layernorm"
        self.network.cam_layer = "convnext.encoder.stages.2.layers.26"
