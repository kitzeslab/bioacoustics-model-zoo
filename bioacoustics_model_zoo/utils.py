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


from github import Github
import os


def download_subfolder(
    repo_full_name,
    subfolder_path,
    local_dir,
    token=None,
    verbose=True,
    redownload_existing=False,
):
    """
    Download a subfolder from a GitHub repo using PyGithub and a custom download_file helper.

    Args:
        repo_full_name (str): e.g., "owner/repo"
        subfolder_path (str): path within repo, e.g., "src/utils"
        local_dir (str): local directory to save files
        download_file (callable): function(url, save_dir, **kwargs) -> None
        token (str, optional): GitHub personal access token
        verbose (bool): print progress
        redownload_existing (bool): force re-download even if file exists
    """
    g = Github(token) if token else Github()
    repo = g.get_repo(repo_full_name)

    def _download_dir_contents(path, local_path):
        os.makedirs(local_path, exist_ok=True)
        contents = repo.get_contents(path)

        for content in contents:
            if content.type == "dir":
                # recurse into subdirectory
                _download_dir_contents(
                    content.path, os.path.join(local_path, content.name)
                )
            else:
                # use download_url directly
                file_save_dir = os.path.join(local_path)
                os.makedirs(file_save_dir, exist_ok=True)
                downloaded_path = download_file(
                    content.download_url,
                    save_dir=file_save_dir,
                    verbose=verbose,
                    redownload_existing=redownload_existing,
                )
                if verbose:
                    print(f"Downloaded {content.path}")

    _download_dir_contents(subfolder_path, local_dir)


def download_cached_subfolder(
    repo_full_name,
    subfolder_path,
    model_name,
    model_version=None,
    cache_dir=None,
    verbose=False,
    redownload_existing=False,
):
    """Download a subfolder of github repo to cache directory if not already cached.

    Args:
        repo_full_name (str): e.g., "owner/repo"
        subfolder_path (str): path within repo of subfolder to download, e.g., "src/utils"
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
    from bioacoustics_model_zoo.cache import get_model_cache_dir

    # Download to cache directory
    model_cache_dir = get_model_cache_dir(
        model_name, model_version=model_version, cache_dir=cache_dir
    )
    save_path = Path(model_cache_dir) / Path(subfolder_path).name
    os.makedirs(save_path, exist_ok=True)
    download_subfolder(
        repo_full_name=repo_full_name,
        subfolder_path=subfolder_path,
        local_dir=str(save_path),
        verbose=verbose,
        redownload_existing=redownload_existing,
    )

    return save_path


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
        """Load samples with specific collate function

        Collate function takes list of AudioSample objects with type(.data)=opensoundscape.Audio
        and returns np.array of shape [batch, length of audio signal]

        Args:
            see SafeAudioDataloader
        """
        kwargs.update({"collate_fn": collate_to_np_array})
        super(AudioSampleArrayDataloader, self).__init__(*args, **kwargs)
