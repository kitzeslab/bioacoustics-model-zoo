# Base configuration. Specific configurations only have to specify the parameters they're changing.
# this file is copied directly from https://github.com/jhuus/HawkEars/blob/0.1.0/core/base_config.py
# and corresponds to tag 0.1.0
from dataclasses import dataclass


@dataclass
class Audio:
    # sampling rate should be a multiple of spec_width / segment_len,
    # so that hop length formula gives an integer (segment_len * sampling_rate / spec_width)
    segment_len = 3  # spectrogram duration in seconds

    spec_height = 192  # spectrogram height
    spec_width = 384  # spectrogram width (3 * 128)
    sampling_rate = 37120
    hop_length = int(segment_len * sampling_rate / spec_width)
    win_length = 2048
    min_audio_freq = 200  # need this low for American Bittern
    max_audio_freq = 13000  # need this high for Chestnut-backed Chickadee "seet series"

    choose_channel = True  # use heuristic to pick the cleanest audio channel
    check_seconds = 3  # check segment of this length when picking cleanest channel
    mel_scale = True
    power = 1.0
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
        False  # usually improves performance, especially with larger models
    )
    multi_label = True
    deterministic = False
    seed = None
    learning_rate = 0.0025  # base learning rate
    batch_size = 64
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
    label_smoothing = 0.125
    training_db = "training"  # name of training database
    num_folds = 1  # for k-fold cross-validation
    val_portion = 0  # used only if num_folds = 1
    model_print_path = "model.txt"  # path of text file to print the model (TODO: put in current log directory)

    # data augmentation (see core/dataset.py to understand these parameters)
    augmentation = True
    prob_simple_merge = 0.35
    prob_real_noise = 0.3
    prob_speckle = 0.1
    prob_fade1 = 0.2
    prob_fade2 = 1
    prob_shift = 1
    max_shift = 6
    min_fade1 = 0.05
    max_fade1 = 0.8
    min_fade2 = 0.1
    max_fade2 = 1
    speckle_variance = 0.012

    classic_mixup = False  # classic mixup is implemented in main_model.py
    classic_mixup_alpha = 1.0


@dataclass
class Inference:
    num_threads = 3  # multiple threads improves performance but uses more GPU memory
    spec_overlap_seconds = (
        1.5  # number of seconds overlap for adjacent 3-second spectrograms
    )
    min_score = 0.75  # only generate labels when score is at least this
    score_exponent = 0.6  # increase scores so they're more like probabilities
    audio_exponent = 0.85  # power parameter for mel spectrograms during inference
    use_banding_codes = True  # use banding codes instead of species names in labels
    top_n = 20  # number of top matches to log in debug mode
    min_location_freq = (
        0.0001  # ignore if species frequency less than this for location/week
    )
    file_date_regex = "\\S+_(\\d+)_.*"  # regex to extract date from file name (e.g. HNCAM015_20210529_161122.mp3)
    file_date_regex_group = 1  # use group at offset 1
    block_size = (
        100  # do this many spectrograms at a time to avoid running out of GPU memory
    )
    frequency_db = "frequency"  # eBird barchart data, i.e. species report frequencies
    all_embeddings = True  # if true, generate embeddings for all spectrograms, otherwise only the labelled ones

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

    # map our species names to the names used by eBird for location/date processing
    ebird_names = {
        "American Goshawk": "Northern Goshawk",
        "Black-crowned Night Heron": "Black-crowned Night-Heron",
    }

    # Low/high/band-pass filters can be used during inference and have to be enabled and configured here.
    # Inference will then use the max prediction per species, with and without the filter(s).
    # Using a single filter adds ~50% to elapsed time for large datasets, but less for small ones where
    # the overhead of loading models etc. is a bigger factor.

    do_unfiltered = True  # set False to run inference with filters only

    # low-pass filter parameters
    do_lpf = False
    lpf_damp = 1  # amount of damping, where 0 does nothing and 1 reduces sounds in the filtered range to 0
    lpf_start_freq = 3500  # start the transition at about this frequency
    lpf_end_freq = 5000  # end the transition at about this frequency

    # high-pass filter parameters
    do_hpf = False
    hpf_damp = 0.9  # amount of damping, where 0 does nothing and 1 reduces sounds in the filtered range to 0
    hpf_start_freq = 2000  # start the transition at about this frequency
    hpf_end_freq = 4000  # end the transition at about this frequency

    # band-pass filter parameters
    do_bpf = False
    bpf_damp = 0.9  # amount of damping, where 0 does nothing and 1 reduces sounds in the filtered range to 0
    bpf_start_freq = 1200  # bottom frequency for band-pass filter is about here
    bpf_end_freq = 7000  # top frequency for band-pass filter is about here


@dataclass
class Miscellaneous:
    main_ckpt_folder = (
        "data/ckpt"  # use an ensemble of all checkpoints in this folder for inference
    )
    low_band_ckpt_path = "data/low_band.ckpt"
    search_ckpt_path = "data/ckpt-search/custom_efficientnet_5.ckpt"  # checkpoint used in searching and clustering
    classes_file = "data/classes.txt"  # list of classes used to generate pickle files
    ignore_file = (
        "data/ignore.txt"  # classes listed in this file are ignored in analysis
    )
    train_pickle = None
    test_pickle = None

    # when running extract and no source is defined, get source by matching these regexes in order;
    # this assumes iNaturalist downloads were renamed by adding an N prefix
    source_regexes = [
        ("XC\\d+", "Xeno-Canto"),
        ("N\\d+", "iNaturalist"),
        ("W\\d+", "Wildtrax"),
        ("HNC.*", "HNC"),
        ("\\d+", "Macaulay Library"),
        (".*", "Other"),
    ]


@dataclass
class BaseConfig:
    audio = Audio()
    train = Training()
    infer = Inference()
    misc = Miscellaneous()
