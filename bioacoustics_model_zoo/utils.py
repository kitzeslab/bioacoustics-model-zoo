import requests
from opensoundscape.ml.dataloaders import SafeAudioDataloader
import numpy as np
from pathlib import Path


def download_github_file(url, save_dir=".", verbose=True, redownload_existing=False):
    save_path = Path(save_dir) / Path(url).name
    if Path(save_path).exists() and not redownload_existing:
        if verbose:
            print(f"File {save_path} already exists; skipping download.")
        return save_path

    # format for github download url:
    # url = f"https://raw.githubusercontent.com/{github_username}/{github_repo}/master/{file_path}"
    # headers = {"Authorization": f"token {github_token}"}
    url = url.replace("/blob/", "/raw/")  # direct download link
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
