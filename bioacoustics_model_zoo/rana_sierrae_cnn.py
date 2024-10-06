from opensoundscape import CNN as _CNN
import torch


def rana_sierrae_cnn(pretrained=True):
    """Load CNN that detects Rana sierrae vocalizations"""

    ## Create model object ##

    # create opensoundscape.CNN object to train a CNN on audio
    model = _CNN(
        architecture="resnet18",
        classes=["rana_sierrae", "negative"],
        sample_duration=2.0,
        single_target=True,
        channels=3,
    )

    ## Preprocessing Parameters ##

    # modify preprocessing of the CNN:
    # bandpass spectrograms to 300-2000 Hz
    model.preprocessor.pipeline.bandpass.set(min_f=300, max_f=2000)

    # use legacy interpolation mode
    model.preprocessor.pipeline.to_tensor.set(use_skimage=True)

    ## Training Parameters ##

    # modify augmentation routine parameters
    model.preprocessor.pipeline.frequency_mask.set(max_masks=5, max_width=0.1)
    model.preprocessor.pipeline.time_mask.set(max_masks=5, max_width=0.1)
    model.preprocessor.pipeline.add_noise.set(std=0.01)

    # decrease the learning rate from the default value
    model.optimizer_params["lr"] = 0.002

    ## Load pre-trained weights ##
    if pretrained:
        dropbox_url = "https://www.dropbox.com/s/9uw1j8yvr75d1dl/BMZ0001_rana_seirrae_cnn_v1-0.model?dl=0"
        download_url = dropbox_url.replace("dropbox.com", "dl.dropboxusercontent.com")
        model.network.load_state_dict(
            torch.hub.load_state_dict_from_url(download_url, progress=False)
        )

    return model
