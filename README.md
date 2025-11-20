# bioacoustics-model-zoo
Pre-trained models for bioacoustic classification tasks

Suggested Citation
> Lapp, S., and Kitzes, J., 2025. "Bioacoustics Model Zoo version 0.12.2". https://github.com/kitzeslab/bioacoustics-model-zoo


## Set up / Installation

### Option 1: Install from PyPI (Recommended)

1. Create a python environment (3.9-3.12 supported) using conda or your preferred package manager:

```bash
conda create -n bmz python=3.11
conda activate bmz
```

2. Install the package with your desired model dependencies:

```bash
# Basic installation (core functionality only)
pip install bioacoustics-model-zoo

# Install with specific model dependencies
pip install bioacoustics-model-zoo[hawkears]  # For HawkEars models
pip install bioacoustics-model-zoo[birdnet]   # For BirdNET model
pip install bioacoustics-model-zoo[tensorflow] # For TensorFlow models (Perch, YAMNet)
pip install bioacoustics-model-zoo[birdset]   # For BirdSet models
pip install bioacoustics-model-zoo[perch]   # For Perch models

# Install all optional dependencies
pip install bioacoustics-model-zoo[all]
```

### Option 2: Install from GitHub (Development)

1. Create a python environment:
```bash
conda create -n bmz python=3.11
conda activate bmz
```

2. Install from GitHub:
```bash
pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo
```

For a specific version:
```bash
pip install git+https://github.com/kitzeslab/bioacoustics-model-zoo@0.12.2
```

3. Install additional dependencies as needed:
```bash
# For HawkEars models
pip install timm torch torchvision torchaudio

# For BirdSet models  
pip install torch torchvision torchaudio transformers

# For TensorFlow models (Perch, YAMNet)
pip install tensorflow tensorflow-hub

# For BirdNET (Note: LiteRT not available on Windows yet)
pip install ai-edge-litert
```

> Note that tensorflow installation sometimes requires careful attention to version numbers, see [this section below](#tensorflow-installation-in-python-environment)

You can now use the package directly in python:
```
import bioacoustics_model_zoo as bmz
model = bmz.HawkEars()
model.predict(audio_files) 
```

See a description of each model and basic usage exmple below. Also see the transfer learning tutorials on OpenSoundscape.org for detailed advice on fine-tuning models from the Bioacoustics Model Zoo. 

If you encounter an issue or a bug, or would like to request a new feature, make a new "Issue" on the [Github Issues page](https://github.com/kitzeslab/bioacoustics-model-zoo/issues). You can also reach out to Sam (`sam.lapp@pitt.edu`) for more specific inquiries. 

# Basic usage

### List: 

List available models in the GitHub repo [bioacoustics-model-zoo](https://github.com/kitzeslab/bioacoustics-model-zoo/)

```python
import bioacoustics_model_zoo as bmz
bmz.list_models() 

# or, for short textual descriptions: 
bmz.describe_models()
```

### Load: 

Get a ready-to-use model object: choose from the models listed in the previous command
```python
model = bmz.BirdNET()
```

### Inference:

Species classification models will have methods for `predict`, `embed`, and `train`.

For instance, use a classification model to generate class presence predictions on an audio file: 

```python
audio_file_path = bmz.birds_path # path to a 10-second audio clip
model = bmz.BirdNET()
scores = model.predict([audio_file_path],activation_layer='softmax')
scores
```

# Model list

### [Perch V2](https://www.kaggle.com/models/google/bird-vocalization-classifier)

Classification and embedding model trained on a large set of annotated bird vocalizations

Preprint:
> van Merriënboer, B., Dumoulin, V., Hamer, J., Harrell, L., Burns, A., & Denton, T. (2025). Perch 2.0: The Bittern Lesson for Bioacoustics. arXiv preprint arXiv:2508.04665.


Perch2 requires tensorflow >=2.20.0. For instance, update to the latest versions using:

```
pip install --upgrade opensoundscape bioacoustics-model-zoo tensorflow tensorflow-hub
```

Example: 

```python
import bioacoustics_model_zoo as bmz
m = bmz.Perch2()
m.predict(['test.wav']) # returns dataframe of per-class scores
m.embed(['test.wav']) # returns dataframe of embeddings

# .forward() returns all outputs: a dictionary with 1D embeddings, 
# spatial embeddings, class scores, and spectrograms
m.forward(['test.wav'],batch_size=32,num_workers=4) 
```

Training: 

The `.train()` method trains a shallow fully-connected neural network as a
classification head while keeping the feature extractor frozen, since the
Perch2 feature extractor is not trainable.

Please see opensoundscape.org documentation and tutorials for detailed walk
through. Once you have multi-hot training and validation label dataframes with
(file, start_time, end_time) multi-index and a column for each class, training
looks like this:

```python
# load the pre-trained Perch2 tensorflow model
m=bmz.Perch2()
# add a 2-layer PyTorch classification head
m.initialize_custom_classifier(classes=train_df.columns, hidden_layer_sizes=(100,))
# embed the training/validation samples with 5 augmented variations each,
# then fit the classification head
m.train(
  train_df,
  val_df,
  n_augmentation_variants=5,
  embedding_batch_size=64,
  embedding_num_workers=4
)
# save the custom Perch2 model to a file
m.save(save_path)
# later, to reload your fine-tuned Perch2 from the saved object:
# m = bmz.Perch2.load(save_path)
```

### [BirdNET](https://github.com/kahst/BirdNET-Analyzer)

Classification and embedding model trained on a large set of annotated bird vocalizations

Additional required packages:

`pip install ai-edge-litert`

Example: 

```python
import bioacoustics_model_zoo as bmz
m = bmz.BirdNET()
m.predict(['test.wav']) # returns dataframe of per-class scores
m.embed(['test.wav']) # returns dataframe of embeddings
```

Training: 

The `.train()` method trains a shallow fully-connected neural network as a
classification head while keeping the feature extractor frozen, since the
BirdNET feature extractor is not open-source. 

Please see opensoundscape.org documentation and tutorials for detailed walk
through. Once you have multi-hot training and validation label dataframes with
(file, start_time, end_time) multi-index and a column for each class, training
looks like this:

```python
# load the pre-trained BirdNET tensorflow model
m=bmz.BirdNET()
# add a 2-layer PyTorch classification head
m.initialize_custom_classifier(classes=train_df.columns, hidden_layer_sizes=(100,))
# embed the training/validation samples with 5 augmented variations each,
# then fit the classification head
m.train(
  train_df,
  val_df,
  n_augmentation_variants=5,
  embedding_batch_size=64,
  embedding_num_workers=4
)
# save the custom BirdNET model to a file
m.save(save_path)
# later, to reload your fine-tuned BirdNET from the saved object:
# m = bmz.BirdNET.load(save_path)
```

### [BirdNET Occurrence Model](https://github.com/kahst/BirdNET-Analyzer)

Predict the probability of observing bird species at a location and week of the year based on eBird occurrence data

Example:


```python
# download and initialize the model
from bioacoustics_model_zoo import BirdNETOccurrenceModel
occurrence_model=BirdNETOccurrenceModel()

# get a dataframe listing species likely to occur at a location and week of the year,
# keeping species only with a relative occurrence probability of 10%
occurrence_model.get_species_list(lat=37.419871, lon=-119.153168,week=34,threshold=.1)
```

### [Perch](https://tfhub.dev/google/bird-vocalization-classifier/4): 

Embedding and bird classification model trained on Xeno Canto

Example:

```python
import bioacoustics_model_zoo as bmz
m = bmz.Perch()
predictions = model.predict(['test.wav']) # predict on the model's classes
embeddings = model.embed(['test.wav']) # generate embeddings on each 5 sec of audio
```

Training: see `BirdNET` example above, training is equivalent (only trains
shallow classifier on frozen feature extractor).

### [HawkEars](https://github.com/jhuus/HawkEars)

Bird classification model for 314 North American species

Note that HawkEars internally uses an ensemble of 5 CNNs. 

Additional required packages:

`timm`, `torchaudio`

Example: 

```python
import bioacoustics_model_zoo as bmz
m = bmz.HawkEars()
m.predict(['test.wav']) # returns dataframe of per-class scores
m.embed(['test.wav']) # returns dataframe of embeddings
```

Training: Training this model is equivalent to training the Opensoundscape.CNN
class. Please see documentation on opensoundscape.org for detailed examples and
walk-throughs. 

Because 5 models are ensembled, training is a bit heavy - you may need small
batch sizes, and you might consider removing all but one model. 

By default, training HawkEars uses a lower learning rate on the feature
extractor than on the classifier - a "fine tuning" paradigm. These values can be modified in the `.optimizer_params` dictionary. 

```python
import bioacoustics_model_zoo as bmz
m = bmz.HawkEars()
m.train(train_df,val_df,epochs=10,batch_size=64,num_workers=4)
```

### [BirdSet ConvNeXT](https://github.com/DBD-research-group/BirdSet)

Open-source PyTorch model trained on Xeno Canto (global bird species classification)

> Rauch, Lukas, et al. "Birdset: A multi-task benchmark for classification in avian bioacoustics." arXiv e-prints (2024): arXiv-2403.


Environment set up:
```bash
conda create -n birdset python=3.10
conda activate birdset
pip install opensoundscape transformers torch torchvision torchaudio
```

Example: predict and embed:
```
import bioacoustics_model_zoo as bmz
m=bmz.BirdSetConvNeXT()
m.predict(['test.wav'],batch_size=64) # returns dataframe of per-class scores
m.embed(['test.wav']) # returns dataframe of embeddings
```

Example: train on different set of classes (see OpenSoundscape tutorials for details on training)
```
import bioacoustics_model_zoo as bmz
import pandas as pd

# load pre-trained network and change output classes
m=bmz.BirdSetConvNeXT()
m.change_classes(['crazy_zebra_grunt','screaming_penguin'])

# optionally, freeze feature extractor (only train final layer)
m.freeze_feature_extractor()

# load one-hot labels and train (index: (file,start_time,end_time))
train_df = pd.read_csv('train_labels.csv',index_col=[0,1,2])
val_df = pd.read_csv('val_labels.csv',index_col=[0,1,2])
m.train(train_df, val_df,batch_size=128, num_workers=8)
```

### [BirdSet EfficientNetB1](https://github.com/DBD-research-group/BirdSet)

Open-source PyTorch model trained on Xeno Canto (global bird species classification)

> Rauch, Lukas, et al. "Birdset: A multi-task benchmark for classification in avian bioacoustics." arXiv e-prints (2024): arXiv-2403.

Environment set up and examples: see BirdSet ConvNeXT, using `m=bmz.BirdSetEfficientNetB1()` 

### [MixIT Bird SeparationModel](https://github.com/google-research/sound-separation/blob/master/models/bird_mixit/README.md)

Separate audio into channels potentially representing separate sources.

This particular model was trained on bird vocalization data. 

Additional required packages:

`tensorflow`, `tensorflow_hub`

Example:

First, download the checkpoint and metagraph from the MixIt Github 
[repo](https://github.com/google-research/sound-separation/blob/master/models/bird_mixit/README.md):
install gsutil then run the following command in your terminal:

`gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .`

Then, use the model in python:
```python
import bioacoustics_model_zoo as bmz
# provide the local path to the checkpoint when creating the object
# this example creates 4 channels; use output_sources8 to separate into 8 channels
model = bmz.SeparationModel(
  checkpoint='/path/to/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090',
)

# separate opensoundscape Audio object into 4 channels:
# note that it seems to work best on 5 second segments
a = Audio.from_file('audio.mp3',sample_rate=22050).trim(0,5)
separated = model.separate_audio(a)

# save audio files for each separated channel:
# saves audio files with extensions like _stem0.wav, _stem1.wav, etc
model.load_separate_write('./temp.wav')
```

### [YAMNet](https://tfhub.dev/google/yamnet/1): 

Embedding model trained on AudioSet YouTube

Additional required packages:

`tensorflow`, `tensorflow_hub`

Example:

```python
import bioacoustics_model_zoo as bmz
m = bmz.YAMNet()
m.predict(['test.wav']) # returns dataframe of per-class scores
m.embed(['test.wav']) # returns dataframe of embeddings
```


### RanaSierraeCNN: 

Detect underwater vocalizations of _Rana sierrae_, the Sierra Nevada Yellow-legged Frog

example: 
```python
import bioacoustics_model_zoo as bmz
m = bmz.RanaSierraeCNN()
m.predict(['test.wav']) # returns dataframe of per-class scores
```

## Other automated detection tools for bioacoustics

### RIBBIT 

Detect sounds with periodic pulsing patterns. 

Implemented in [OpenSoundscape](https://opensoundscape.org) as
`opensoundscape.ribbit.ribbit()`.

- [Python notebooks demonstrating use](https://github.com/kitzeslab/ribbit_manuscript_notebooks)
- [Implementation for R](https://github.com/kitzeslab/r-ribbit)
- [Manuscript: Lapp et al 2021](https://conbio.onlinelibrary.wiley.com/doi/full/10.1111/cobi.13718)

### Accelerating and decelerating sequences

Detect pulse trains that accelerate, such as the drumming of Ruffed Grouse (_Bonasa umbellus_)

Implemented in [OpenSoundscape](https://opensoundscape.org) as

`opensoundscape.signal_processing.detect_peak_sequence_cwt()`. 

(note that in earlier versions of 
OpenSoundscape the module is named `signal` rather than `signal_processing`)

- [Python notebooks demonstrating use](https://github.com/kitzeslab/ruffed_grouse_manuscript_2022)
- [Manuscript: Lapp et al 2022](https://wildlife.onlinelibrary.wiley.com/doi/full/10.1002/wsb.1395)


# Converting BMZ models to Pytorch Lightning + Opensoundscape
Opensoundscape 0.13.0 will make this easier by implementing LightningSpectrogramModule.from_model(model)

```python
import opensoundscape as opso # v0.12.0
import bioacoustics_model_zoo as bmz
from opensoundscape.ml.lightning import LightningSpectrogramModule

# Load any Pytorch-based model from the model zoo (not TensorFlow models like Perch/BirdNET)
model = bmz.BirdSetConvNeXT()

#convert to Lightning model object with .predict_with_trainer, .fit_with_trainer methods
# 
lm = LightningSpectrogramModule(
    architecture=model.network, classes=model.classes, sample_duration=model.preprocessor.sample_duration
)
lm.preprocessor = model.preprocessor
# lm.predict_with_trainer(opso.birds_path)
# lm.fit_with_trainer(...)
```

# Contributing

To contribute a model to the model zoo, email `sam.lapp@pitt.edu` or add a model yourself:
- fork this repository ([help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo))
- add a `.py` module in the bioacoustics_model_zoo subfolder implementing a class that instantiates your model object
  - implement the predict() and embed() methods with an API matching the other models in the model zoo
  - optionally implement train() method
  - Note: if you have a pytorch model, you may be able to simply subclass opensoundscape.CNN without needing to override these methods
  - in the docstring, provide an example of use
  - in the docstring, also include a suggested citation for others using the model
  - decorate your class with `@register_bmz_model` 
- add an import statement in `__init__.py` to import your model class into the top-level package API (`from bioacoustics_model_zoo.new_model import NewModel`)
- add your model to the Model List below in this document, with example usage
- submit a pull request ([GitHub's help page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork))

Check out any of the existing models for examples of how to complete these steps. In particular, pick the current model class most similar to yours (pytorch vs tensorflow) as a starting point. 


# Troubleshooting 

### TensorFlow Installation in Python Environment

Some models in the model zoo require tensorflow (and potentially tensorflow_hub) to be installed in your python environment. 

Installing TensorFlow can be tricky, and it may not be possible to have cuda-enabled tensorflow in the same environment as cuda-enabled pytorch. In this case, you can install a cpu-only version of tensorflow (`pip install tensorflow-cpu`). You may want to start with a fresh environment, or uninstall tensorflow and nvidia-cudnn-cu11 then reinstall pytorch with the appropriate nvidia-cudnn-cu11, to avoid having the wrong cudnn for PyTorch. 

Alternatively, if you want to use the TensorFlow Hub models with GPU acceleration, create an environment where you uninstall `pytorch` and `nvidia-cudnn-cu11` and install a cpu-only version (see [this page](https://pytorch.org/get-started/locally/) for the correct installation command). Then, you can `pip install tensorflow-hub` and let it choose the correct nvidia-cudnn so that it can use CUDA and leverage GPU acceleration. 

Installing tensorflow: Carefully follow the [directions](https://www.tensorflow.org/install/pip) for your system. Note that models provided in this repo might require the specific nvidia-cudnn-cu11 version 8.6.0, which could conflict with the version required for pytorch. 

### Error while Downloading TF Hub Models

Some of the models provided in this repo are hosted on the Tensorflow model hub. 

If you encounter the following error (or similar) when downloading a TensorFlow Hub model:

```
ValueError: Trying to load a model of incompatible/unknown type. '/var/folders/d8/265wdp1n0bn_r85dh3pp95fh0000gq/T/tfhub_modules/9616fd04ec2360621642ef9455b84f4b668e219e' contains neither 'saved_model.pb' nor 'saved_model.pbtxt'.
```

You need to delete the folder listed in the error message (something like `/var/folders/...tfhub_modules/....`). After deleting that folder, downloading the model should work. 

The issue occurs because TensorFlow Hub is looking for a cached 
model in a temporary folder where it was once stored but no longer exists. See relevant GitHub issue here: 
https://github.com/tensorflow/hub/issues/896
