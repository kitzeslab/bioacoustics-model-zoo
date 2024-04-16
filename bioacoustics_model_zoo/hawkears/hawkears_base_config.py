# Base configuration. Specific configurations only have to specify the parameters they"re changing.

from dataclasses import dataclass


@dataclass
class Audio:
    segment_len = 3  # spectrogram duration in seconds
    sampling_rate = 40960
    hop_length = 320
    win_length = 2048
    spec_height = 192  # spectrogram height
    spec_width = 384  # spectrogram width (3 * 128)
    choose_channel = True  # use heuristic to pick the cleanest audio channel
    check_seconds = 3  # check prefix of this length when picking cleanest channel
    min_audio_freq = 200  # need this low for American Bittern
    max_audio_freq = 13000  # need this high for Chestnut-backed Chickadee "seet series"
    mel_scale = True
    power = 1
    spec_block_seconds = (
        240  # max seconds of spectrogram to create at a time (limited by GPU memory)
    )

    # low-frequency audio settings for Ruffed Grouse drumming identifier
    low_band_spec_height = 64
    low_band_min_audio_freq = 0
    low_band_max_audio_freq = 200
    low_band_mel_scale = False


@dataclass
class Training:
    compile = False
    mixed_precision = (
        True  # usually improves performance, especially with larger models
    )
    multi_label = True
    deterministic = False
    seed = None
    learning_rate = 0.0025  # base learning rate
    batch_size = 32
    model_name = "tf_efficientnetv2_b0"  # 5.9M parameters
    load_weights = False  # passed as "weights" to timm.create_model
    use_class_weights = True
    load_ckpt_path = None  # for transfer learning or fine-tuning
    update_classifier = False  # if true, create a new classifier for the loaded model
    freeze_backbone = False  # if true, freeze the loaded model and train the classifier only (requires update_classifier=True)
    num_workers = 2
    dropout = None  # various dropout parameters are passed to model only if not None
    drop_rate = None
    drop_path_rate = None
    proj_drop_rate = None
    num_epochs = 10
    LR_epochs = (
        None  # default = num_epochs, higher values reduce effective learning rate decay
    )
    save_last_n = 3  # save checkpoints for this many last epochs
    label_smoothing = 0.15
    training_db = "training"  # name of training database
    num_folds = 1  # for k-fold cross-validation
    val_portion = 0  # used only if num_folds > 1
    model_print_path = "model.txt"  # path of text file to print the model (TODO: put in current log directory)

    # data augmentation (see core/dataset.py to understand these parameters)
    augmentation = True
    prob_mixup = 0.35
    prob_real_noise = 0.3
    prob_speckle = 0.1
    prob_fade = 0.2
    prob_exponent = 0.25
    prob_shift = 1
    max_shift = 6
    min_fade = 0.1
    max_fade = 0.8
    speckle_variance = 0.009
    min_exponent = 1
    max_exponent = 1.6
    mixup_weights = False
    mixup_weight_min = 0.2
    mixup_weight_max = 0.8

    augmix = False
    augmix_factor = 4


@dataclass
class Inference:
    num_threads = 3  # multiple threads improves performance but uses more GPU memory
    spec_overlap_seconds = (
        1.5  # number of seconds overlap for adjacent 3-second spectrograms
    )
    min_score = 0.7  # only generate labels when score is at least this
    score_exponent = 0.6  # increase scores so they're more like probabilities
    use_banding_codes = True  # use banding codes instead of species names in labels
    top_n = 20  # number of top matches to log in debug mode
    min_location_freq = (
        0.0001  # ignore if species frequency less than this for location/week
    )
    file_date_regex = "\S+_(\d+)_.*"  # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
    file_date_regex_group = 1  # use group at offset 1
    block_size = (
        100  # do this many spectrograms at a time to avoid running out of GPU memory
    )
    frequency_db = "frequency"  # eBird barchart data, i.e. species report frequencies

    # These parameters control a second pass during inference.
    # If lower_min_if_confirmed is true, count the number of seconds for a species in a recording,
    # where score >= min_score + raise_min_to_confirm * (1 - min_score).
    # If seconds >= confirmed_if_seconds, the species is assumed to be present, so scan again,
    # lowering the min_score by multiplying it by lower_min_factor.
    lower_min_if_confirmed = True
    raise_min_to_confirm = (
        0.5  # to be confirmed, score must be >= min_score + this * (1 - min_score)
    )
    confirmed_if_seconds = (
        8  # need at least this many confirmed seconds >= raised threshold
    )
    lower_min_factor = 0.6  # if so, include all labels with score >= this * min_score


@dataclass
class Miscellaneous:
    main_ckpt_folder = (
        "data/ckpt"  # use an ensemble of all checkpoints in this folder for inference
    )
    low_band_ckpt_path = "data/low_band.ckpt"
    search_ckpt_path = "data/ckpt/custom_efficientnet_5B.ckpt"  # checkpoint used in searching and clustering
    classes_file = "data/classes.txt"  # list of classes used to generate pickle files
    ignore_file = (
        "data/ignore.txt"  # classes listed in this file are ignored in analysis
    )
    train_pickle = None
    test_pickle = None

    # when running extract and no source is defined, get source by matching these regexes in order;
    # this assumes iNaturalist downloads were renamed by adding an N prefix
    source_regexes = [
        ("XC\d+", "Xeno-Canto"),
        ("N\d+", "iNaturalist"),
        ("W\d+", "Wildtrax"),
        ("HNC.*", "HNC"),
        ("\d+", "Macaulay Library"),
        (".*", "Other"),
    ]


@dataclass
class BaseConfig:
    audio = Audio()
    train = Training()
    infer = Inference()
    misc = Miscellaneous()
