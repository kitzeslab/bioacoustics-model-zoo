from opensoundscape.preprocess.preprocessors import AudioAugmentationPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader
from bioacoustics_model_zoo.utils import collate_to_np_array

from pathlib import Path
from bioacoustics_model_zoo.utils import (
    download_cached_subfolder,
    register_bmz_model,
    download_cached_file,
)

import numpy as np
from tqdm import tqdm

import pandas as pd


@register_bmz_model
class Nighthawk:
    """
    Nighthawk model for bird Nocturnal Flight Call detection

    NightHawk is currently inference-only, and does not support training, embedding, or fine-tuning.

    Exmple:
    ```python
    from bioacoustics_model_zoo import Nighthawk
    model = Nighthawk()
    results = model.predict("path/to/audio.wav")
    results["species"]  # pandas DataFrame of species-level predictions
    ```
    """

    def __init__(self, version="v0.3.1", cache_dir=None, redownload_existing=False):
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "Tensorflow is required to use this model and was not found in the environment"
            )

        # download model files:
        model_dir = download_cached_subfolder(
            "bmvandoren/Nighthawk",
            "nighthawk/saved_model_with_preprocessing",
            model_name="nighthawk",
            model_version=version,
            cache_dir=cache_dir,
            redownload_existing=redownload_existing,
        )

        # download taxonomy / class lists
        github_url = f"https://github.com/bmvandoren/Nighthawk/raw/refs/tags/{version}/nighthawk/"
        self.classes = {}
        for file_name in ["species", "groups", "families", "orders"]:
            local_file = download_cached_file(
                f"{github_url}/taxonomy/{file_name}.txt",
                filename=None,
                model_name="nighthawk",
                model_version=version,
            )
            self.classes[file_name] = pd.read_csv(local_file, header=None).values[:, 0]

        sp_group_table = download_cached_file(
            f"{github_url}/taxonomy/groups_ebird_codes.csv",
            model_name="nighthawk",
            filename=None,
            model_version=version,
        )
        self.species_to_groups = pd.read_csv(sp_group_table)

        self.tf_model = tf.saved_model.load(model_dir)
        self.preprocessor = AudioAugmentationPreprocessor(
            sample_duration=1, sample_rate=22050, extend_short_clips=True
        )
        # this is already the default, but be explicit in case the default changes
        self.preprocessor.pipeline.load_audio.set(resample_type="soxr_hq")

    def predict_dataloader(self, samples, **kwargs):
        """generate dataloader for inference (predict/validate/test)

        Args: see self.inference_dataloader_cls docstring for arguments
            **kwargs: any arguments to pass to the DataLoader __init__
            Note: these arguments are fixed and should not be passed in kwargs:
            - shuffle=False: retain original sample order
        """
        # for convenience, convert str/pathlib.Path to list of length 1
        if isinstance(samples, (str, Path)):
            samples = [samples]

        return SafeAudioDataloader(
            samples=samples,
            preprocessor=self.preprocessor,
            shuffle=False,  # keep original order
            pin_memory=False,  # if self.device == torch.device("cpu") else True,
            collate_fn=collate_to_np_array,
            **kwargs,
        )

    def __call__(self, dataloader, progress_bar=True):
        # appears to not support batch inference, so we predict on one sample at a time
        predictions = []
        for batch_data, _ in tqdm(dataloader, disable=not progress_bar):
            batch_data = batch_data.astype(np.float32)
            predictions.extend([self.tf_model(s) for s in batch_data])

        # collate
        # Put order, family, group and species logit tensors into their
        # own two-dimensional NumPy arrays, squeezing out the first tensor
        # dimension, which always has length one. The result is a list of four
        # two dimensional NumPy arrays, one each for order, family,
        # group, and species. The first index of each array is for input
        # and the second is for logit.
        predictions = [np.squeeze(np.array(p), axis=1) for p in zip(*predictions)]
        return {
            "order": predictions[0],
            "family": predictions[1],
            "group": predictions[2],
            "species": predictions[3],
        }

    def predict(self, samples, progress_bar=True, **kwargs):
        dataloader = self.predict_dataloader(samples, **kwargs)
        preds = self(dataloader, progress_bar=progress_bar)
        # create dataframes
        results = {}
        index = dataloader.dataset.dataset.label_df.index
        results["order"] = pd.DataFrame(
            preds["order"], columns=self.classes["orders"], index=index
        )
        results["family"] = pd.DataFrame(
            preds["family"], columns=self.classes["families"], index=index
        )
        results["group"] = pd.DataFrame(
            preds["group"], columns=self.classes["groups"], index=index
        )
        results["species"] = pd.DataFrame(
            preds["species"], columns=self.classes["species"], index=index
        )
        return results
