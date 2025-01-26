dependencies = ["torch", "opensoundscape"]
import torch

# handle optional dependencies
try:
    import tensorflow as tf
except:
    # allow use without tensorflow
    tf = None

try:
    import timm
except:
    timm = None

try:
    import torchaudio
except:
    torchaudio = None

# import with leading underscore to hide from torch.hub.list()
from opensoundscape import CNN as _CNN

# each function we import will be visible in torch.hub.list()

# pytorch models
from bioacoustics_model_zoo.rana_sierrae_cnn import RanaSierraeCNN


class MissingTFDependency:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Tensorflow is required to use this model and was not found in the environment"
        )


# tensorflow requirement
if tf is None:

    class BirdNET(MissingTFDependency):
        pass

    class SeparationModel(MissingTFDependency):
        pass

    class YAMNet(MissingTFDependency):
        pass

    class Perch(MissingTFDependency):
        pass

else:
    from bioacoustics_model_zoo.birdnet import BirdNET
    from bioacoustics_model_zoo.mixit_separation import SeparationModel
    from bioacoustics_model_zoo.yamnet import YAMNet
    from bioacoustics_model_zoo.perch import Perch


# timm and torchaudio requirement
class MissingHawkearsDependency:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "timm and torchaudio packages are required to use this model and at least one was not found in the environment"
        )


if timm is None or torchaudio is None:

    class HawkEars(MissingHawkearsDependency):
        pass

else:
    from bioacoustics_model_zoo.hawkears.hawkears import HawkEars

## see instructions here for hubconf file:
## https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url

# to create direct download links for OneDrive, follow these instructions:
# https://learn.microsoft.com/en-us/graph/api/shares-get?view=graph-rest-1.0&tabs=http#encoding-sharing-urls
