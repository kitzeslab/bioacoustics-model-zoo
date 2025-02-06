import requests
from opensoundscape.ml.dataloaders import SafeAudioDataloader
import numpy as np
from pathlib import Path


def register_model(cls):
    """Register a model class to be visible in bmz.list_models()

    Args:
        cls: class to register
    """


global BMZ_MODEL_LIST
BMZ_MODEL_LIST = []


def list_models():
    """return list of available action function keyword strings
    (can be used to initialize Action class)
    """
    return BMZ_MODEL_LIST


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


def download_github_file(url, save_dir=".", verbose=True, redownload_existing=False):
    save_path = Path(save_dir) / Path(url).name
    if Path(save_path).exists() and not redownload_existing:
        if verbose:
            print(f"File {save_path} already exists; skipping download.")
        return save_path

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
