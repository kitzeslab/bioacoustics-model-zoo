import requests
from opensoundscape.ml.dataloaders import SafeAudioDataloader
import numpy as np
from pathlib import Path


def register_model(cls):
    """Register a model class to be visible in bmz.list_models()

    Args:
        cls: class to register
    """


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


def download_cached_file(
    url,
    filename,
    model_name,
    model_version=None,
    cache_dir=None,
    verbose=False,
    redownload_existing=False,
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
        if verbose:
            print(f"File {filename} found in cache at {cached_file_path}")
        return cached_file_path

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
