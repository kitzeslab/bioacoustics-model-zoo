__version__ = "0.12.1"

import torch
from bioacoustics_model_zoo import utils
from bioacoustics_model_zoo.utils import list_models, describe_models, BMZ_MODEL_LIST
from bioacoustics_model_zoo.utils import register_bmz_model
from bioacoustics_model_zoo import cache
from bioacoustics_model_zoo.cache import (
    get_default_cache_dir,
    set_default_cache_dir,
    clear_cached_model,
    clear_all_cached_models,
)

# import a sample audio file and path to that file in the top-level API
from opensoundscape import birds, birds_path

# handle optional dependencies
try:
    import tensorflow as tf
except:
    # allow use without tensorflow
    tf = None

try:
    import ai_edge_litert
except:
    # allow use without tflite
    ai_edge_litert = None

try:
    import timm
except:
    timm = None

try:
    import torchaudio
except:
    torchaudio = None

try:
    import transformers
except:
    transformers = None

# import with leading underscore to hide from torch.hub.list()
from opensoundscape import CNN as _CNN

# each function we import will be visible in bmz.utils.list_models()


class MissingTFDependency:
    """Tensorflow dependency missing! try `pip install tensorflow`"""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "Tensorflow is required to use this model and was not found in the environment"
        )


class MissingTFLiteDependency:
    """ai_edge_litert (tflite) dependency missing! try `pip install ai-edge-litert`"""

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "ai_edge_litert (tflite) is required to use this model and was not found in the environment"
        )


# tflite requirement
if ai_edge_litert is None and tensorflow is None:

    @register_bmz_model
    class BirdNET(MissingTFLiteDependency):

        pass

    @register_bmz_model
    class BirdNETOccurrenceModel(MissingTFLiteDependency):

        pass

else:
    from bioacoustics_model_zoo.birdnet import BirdNET, BirdNETOccurrenceModel

# tensorflow requirement
if tf is None:

    @register_bmz_model
    class SeparationModel(MissingTFDependency):
        pass

    @register_bmz_model
    class YAMNet(MissingTFDependency):
        pass

    @register_bmz_model
    class Perch(MissingTFDependency):
        pass

    @register_bmz_model
    class Perch2(MissingTFDependency):
        pass

else:
    from bioacoustics_model_zoo.mixit_separation import SeparationModel
    from bioacoustics_model_zoo.yamnet import YAMNet
    from bioacoustics_model_zoo.perch import Perch
    from bioacoustics_model_zoo.perch_v2 import Perch2
    from bioacoustics_model_zoo import birdnet
    from bioacoustics_model_zoo import mixit_separation
    from bioacoustics_model_zoo import yamnet
    from bioacoustics_model_zoo import perch, perch_v2


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

    @register_bmz_model
    class HawkEars_Embedding(MissingHawkearsDependency):
        pass

    @register_bmz_model
    class HawkEars_Low_Band(MissingHawkearsDependency):
        pass

    @register_bmz_model
    class HawkEars_v010(MissingHawkearsDependency):
        pass

else:
    from bioacoustics_model_zoo.hawkears.hawkears import (
        HawkEars,
        HawkEars_Embedding,
        HawkEars_Low_Band,
        HawkEars_v010,
    )
    from bioacoustics_model_zoo import hawkears


class MissingBirdSetDependency:
    """BirdSetConvNeXT dependency missing!

    try:
    pip install torch torchaudio torchvision transformers
    """

    def __init__(self, *args, **kwargs):
        raise ImportError(
            """BirdSetConvNeXT dependency missing!

            try:
            pip install torch torchaudio torchvision transformers
            """
        )


if transformers is None or torchaudio is None:

    @register_bmz_model
    class BirdSetConvNeXT(MissingBirdSetDependency):
        pass

    @register_bmz_model
    class BirdSetEfficientNetB1(MissingBirdSetDependency):
        pass

else:
    from bioacoustics_model_zoo import bmz_birdset
    from bioacoustics_model_zoo.bmz_birdset import (
        bmz_birdset_convnext,
        bmz_birdset_efficientnetB1,
        birdset_preprocessing,
    )
    from bioacoustics_model_zoo.bmz_birdset.bmz_birdset_convnext import BirdSetConvNeXT
    from bioacoustics_model_zoo.bmz_birdset.bmz_birdset_efficientnetB1 import (
        BirdSetEfficientNetB1,
    )

from bioacoustics_model_zoo import rana_sierrae_cnn
from bioacoustics_model_zoo.rana_sierrae_cnn import RanaSierraeCNN
