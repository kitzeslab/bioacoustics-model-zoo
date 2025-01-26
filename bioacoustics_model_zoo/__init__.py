import torch
import utils
from utils import register_bmz_model

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

# each function we import will be visible in bmz.utils.list_models()


class MissingTFDependency:
    """Tensorflow dependency missing! try `pip install tensorflow`"""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Tensorflow is required to use this model and was not found in the environment"
        )


# tensorflow requirement
if tf is None:

    @register_bmz_model
    class BirdNET(MissingTFDependency):

        pass

    @register_bmz_model
    class SeparationModel(MissingTFDependency):
        pass

    @register_bmz_model
    class YAMNet(MissingTFDependency):
        pass

    @register_bmz_model
    class Perch(MissingTFDependency):
        pass

else:
    from bioacoustics_model_zoo.birdnet import BirdNET
    from bioacoustics_model_zoo.mixit_separation import SeparationModel
    from bioacoustics_model_zoo.yamnet import YAMNet
    from bioacoustics_model_zoo.perch import Perch
    from . import birdnet
    from . import mixit_separation
    from . import yamnet
    from . import perch


# timm and torchaudio requirement
class MissingHawkearsDependency:
    """HawkEars dependency missing! try `pip install timm torchaudio`"""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "timm and torchaudio packages are required to use this model and at least one was not found in the environment"
        )


if timm is None or torchaudio is None:

    @register_bmz_model
    class HawkEars(MissingHawkearsDependency):
        pass

else:
    from bioacoustics_model_zoo.hawkears.hawkears import HawkEars
    from . import hawkears

from . import rana_sierrae_cnn
from .rana_sierrae_cnn import RanaSierraeCNN
