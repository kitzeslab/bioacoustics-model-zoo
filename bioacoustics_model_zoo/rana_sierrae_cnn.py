from opensoundscape import CNN as _CNN
from opensoundscape.ml.cnn_architectures import resnet18
import torch
from bioacoustics_model_zoo.utils import register_bmz_model


@register_bmz_model
class RanaSierraeCNN(_CNN):
    def __init__(self):
        """CNN trained to detect Rana sierrae calls

        This model has two classes 'rana_sierrae' and 'negative', and predicts
        the presence of Rana sierrae (Sierra Nevada Yellow-legged frog) calls in
        underwater audio recordings.

        Downloads the trained model weights from a public dropbox link.

        This model is described in the following publication:

        > Lapp, Sam, et al. "Aquatic Soundscape Recordings Reveal Diverse
        Vocalizations and Nocturnal Activity of an Endangered Frog." The
        American Naturalist 203.5 (2024): 618-627.

        """
        # initialize resnet with random weights, since we will load pre-trained weights
        arch = resnet18(num_classes=2, weights=None)
        super().__init__(
            architecture=arch,
            classes=["rana_sierrae", "negative"],
            sample_duration=2.0,
            single_target=True,
            channels=3,
        )

        # modify preprocessing of the CNN:
        # bandpass spectrograms to 300-2000 Hz
        self.preprocessor.pipeline.bandpass.set(min_f=300, max_f=2000)

        # use legacy interpolation mode
        self.preprocessor.pipeline.to_tensor.set(use_skimage=True)

        # modify augmentation routine parameters
        self.preprocessor.pipeline.frequency_mask.set(max_masks=5, max_width=0.1)
        self.preprocessor.pipeline.time_mask.set(max_masks=5, max_width=0.1)
        self.preprocessor.pipeline.add_noise.set(std=0.01)

        # decrease the learning rate from the default value
        self.optimizer_params["lr"] = 0.002

        ## Load pre-trained weights ##
        dropbox_url = "https://www.dropbox.com/s/9uw1j8yvr75d1dl/BMZ0001_rana_seirrae_cnn_v1-0.model?dl=0"
        download_url = dropbox_url.replace("dropbox.com", "dl.dropboxusercontent.com")
        self.network.load_state_dict(
            torch.hub.load_state_dict_from_url(download_url, progress=False)
        )
