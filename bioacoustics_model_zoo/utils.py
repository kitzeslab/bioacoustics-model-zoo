import requests
from opensoundscape.ml.dataloaders import SafeAudioDataloader
import numpy as np
from pathlib import Path
import torch
import hashlib

BMZ_MODEL_LIST = []


def list_models():
    """return dictionary of available model names and classes"""
    global BMZ_MODEL_LIST
    return {c.__name__: c for c in BMZ_MODEL_LIST}


def describe_models():
    """return short description of each available model"""
    descriptions = {}
    for model in BMZ_MODEL_LIST:
        if model.__doc__ is not None:
            txt = model.__doc__.split("\n")[0]
        elif model.__init__.__doc__ is not None:
            txt = model.__init__.__doc__.split("\n")[0]
        else:
            txt = "no description"
        descriptions[model.__name__] = txt

    return descriptions


def register_bmz_model(model_cls):
    """add class to BMZ_MODEL_LIST

    this allows us to recreate the class when loading saved model file with load_model()
    """
    # register the model in dictionary
    BMZ_MODEL_LIST.append(model_cls)
    # return the function
    return model_cls


def download_file(url, save_dir=".", verbose=False, redownload_existing=False):
    save_path = Path(save_dir) / Path(url).name
    if Path(save_path).exists() and not redownload_existing:
        if verbose:
            print(f"File {save_path} already exists; skipping download.")
        return save_path

    if "github.com" in url:
        # format for github download url:
        # url = f"https://raw.githubusercontent.com/{github_username}/{github_repo}/master/{file_path}"
        # headers = {"Authorization": f"token {github_token}"}
        url = str(url).replace("/blob/", "/raw/")  # direct download link
    response = requests.get(url)  # , headers=headers)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        if verbose:
            print(f"Downloaded completed: {Path(url).name}")
    else:
        raise Exception(
            f"Failed to download file from url {url}. Status code: {response.status_code}"
        )

    return save_path


def verify_file_sha256(file_path: str, expected_sha256: str) -> bool:
    """Computes the SHA256 hash of a file and verifies it against an expected string."""
    sha256_hash = hashlib.sha256()

    # Open the file in binary read mode ('rb')
    with open(file_path, "rb") as f:
        # Read the file in 64KB blocks to prevent memory crashes on large files
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)

    # Get the hex string format of the calculated hash
    calculated_sha256 = sha256_hash.hexdigest()

    # Strip whitespace and compare case-insensitively
    return calculated_sha256.lower() == expected_sha256.strip().lower()


def download_cached_file(
    url,
    filename,
    model_name,
    model_version=None,
    cache_dir=None,
    verbose=False,
    redownload_existing=False,
    sha256=None,
):
    """Download a file to cache directory if not already cached.

    Args:
        url (str): URL to download from
        filename (str): Name to save file as (extracted from URL if None)
        model_name (str): Name of the model for cache organization
        model_version (str): Version of the model for cache organization
            - if specified, does not consider model to be cached if version mismatches
            and stores model in [cache_dir]/[model_name]/[model_version]/
            - if not specified, assumes model has not changed versions and
            stores model in [cache_dir]/[model_name]/
        cache_dir (str or Path, optional): Override cache directory
        verbose (bool): Print download messages
        redownload_existing (bool): Re-download even if file exists
        sha256 (str, optional): Expected SHA256 hash of the downloaded file
            - if file exists and hash does not match, re-downloads the file
    Returns:
        Path: Path to the cached file
    """
    from bioacoustics_model_zoo.cache import (
        get_cached_file_path,
        is_cached,
        get_model_cache_dir,
    )

    if filename is None:
        filename = Path(url).name

    cached_file_path = get_cached_file_path(
        filename=filename,
        model_name=model_name,
        model_version=model_version,
        cache_dir=cache_dir,
    )

    # Check if file already exists in cache
    if (
        is_cached(
            filename=filename,
            model_name=model_name,
            model_version=model_version,
            cache_dir=cache_dir,
        )
        and not redownload_existing
    ):
        # If sha256 is provided, verify the hash
        skip_download = verify_file_sha256(cached_file_path, sha256) if sha256 else True
        if skip_download:
            if verbose:
                print(f"File {filename} found in cache at {cached_file_path}")
            return cached_file_path
        else:
            if verbose:
                print(
                    f"File {filename} found in cache but SHA256 hash does not match. Re-downloading."
                )
            # remove the existing file with incorrect hash
            # Path(cached_file_path).unlink()

    # Download to cache directory
    model_cache_dir = get_model_cache_dir(
        model_name, model_version=model_version, cache_dir=cache_dir
    )
    downloaded_path = download_file(
        url,
        save_dir=str(model_cache_dir),
        verbose=verbose,
        redownload_existing=redownload_existing,
    )

    return Path(downloaded_path)


def collate_to_np_array(audio_samples):
    """
    takes list of AudioSample objects with type(sample.data)==opensoundscape.Audio
    and returns (samples, labels);
        - samples is np.array of shape [batch, length of audio signal]
        - labels is np.array of shape [batch, n_classes]
    """
    try:
        return (
            np.array([a.data.samples for a in audio_samples]),
            np.vstack([a.labels.values for a in audio_samples]),
        )
    except Exception as exc:
        raise ValueError(
            "Must pass list of AudioSample with Audio object as .data"
        ) from exc


class AudioSampleArrayDataloader(SafeAudioDataloader):
    def __init__(self, *args, **kwargs):
        """Load audio samples, collating to np.array of audio signals unless collate_fn is specified.

        Collate function takes list of AudioSample objects with type(.data)=opensoundscape.Audio
        and returns np.array of shape [batch, length of audio signal]

        Args:
            see SafeAudioDataloader
        """
        if not "collate_fn" in kwargs or kwargs["collate_fn"] is None:
            kwargs.update({"collate_fn": collate_to_np_array})
        super(AudioSampleArrayDataloader, self).__init__(*args, **kwargs)


class Ensemble(torch.nn.Module):
    """Ensemble of multiple models for classification

    Args:
        models: list of models to use in the ensemble
    """

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        for i, m in enumerate(models):
            self.add_module(f"model_{i}", m)

    def forward(self, x):
        """forward pass through the ensemble, average outputs from across models"""
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)
