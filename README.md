# bioacoustics-model-zoo
Pre-trained models for bioacoustic classification tasks

## Set up / Installation
To use the bioacoustics model zoo: 
1. create a python environment (3.9-3.11 supported) using conda or your preferred package manager
2. install the required packages: download the requirements.txt file, then run `pip install -r /path/to/requirements.txt`
3. use the torch.hub API to access models:
```python
import torch
model = torch.hub.load('kitzeslab/bioacoustics-model-zoo','Perch')
```

or the opensoundscape API:

```python
from opensoundscape.ml import bioacoustics_model_zoo as bmz
model = bmz.load('Perch')
```

# Basic usage

### List: 

List available models in the GitHub repo [bioacoustics-model-zoo](https://github.com/kitzeslab/bioacoustics-model-zoo/)
```
import torch
torch.hub.list('kitzeslab/bioacoustics-model-zoo')
```

### Load: 

Get a ready-to-use model object: choose from the models listed in the previous command
```
model = torch.hub.load('kitzeslab/bioacoustics-model-zoo','rana_sierrae_cnn',trust_repo=True)
```

### Inference:

`model` is an OpenSoundscape CNN object (or other class) which you can use as normal. 

For instance, use the model to generate predictions on an audio file: 

```
audio_file_path = './hydrophone_10s.wav'
scores = model.predict([audio_file_path],activation_layer='softmax')
scores
```

# Contributing

To contribute a model to the model zoo, email `sam.lapp@pitt.edu` or add a model yourself:
- fork this repository ([help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo))
- add a .py module in the bioacoustics_model_zoo subfolder implementing a class that instantiates your model object
  - your trained model object or weights will not be saved in this repository. The class should instead load the pre-trained weights from a public url. For an example:
https://github.com/kitzeslab/bioacoustics-model-zoo/blob/593fe39a9e0f712c04d6e20262af70aeac20be75/hubconf.py#L53-L57
  - include a thorough docstring describing the model's purpose, how it was trained, and how to use it
  - in the docstring, also include a suggested citation for others using the model
- add a line to `hubconf.py` importing your class, eg `from bioacoustics_model_zoo.birdnet import BirdNET`
- add your model to the Model List below in this document, with example usage
- submit a pull request ([GitHub's help page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork))

  

# Model list

### [Perch](https://tfhub.dev/google/bird-vocalization-classifier/4): 

Embedding and bird classification model trained on Xeno Canto

Example:

```python
import torch
model = torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'Perch',trust_repo=True)
predictions = model.predict(['test.wav']) # predict on the model's classes
embeddings = model.generate_embeddings(['test.wav']) # generate embeddings on each 5 sec of audio
```

### [BirdNET](https://github.com/kahst/BirdNET-Analyzer)

Classification and embedding model trained on a large set of annotated bird vocalizations

Additional required packages:

`tensorflow`, `tensorflow_hub`

Example: 

```python
import torch
m = torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'BirdNET',trust_repo=True)
m.predict(['test.wav']) # returns dataframe of per-class scores
m.generate_embeddings(['test.wav']) # returns dataframe of embeddings
```

### [HawkEars](https://github.com/jhuus/HawkEars)

Bird classification CNN for 314 North American species

Additional required packages:

`timm`, `torchaudio`

Example: 
```python
import torch
m = torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'HawkEars',trust_repo=True)
m.predict(['test.wav']) # returns dataframe of per-class scores
```

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
import torch
# provide the local path to the checkpoint when creating the object
model = torch.hub.load(
    'kitzeslab/bioacoustics-model-zoo',
    'SeparationModel',
    checkpoint='/path/to/bird_mixit_model_checkpoints/output_sources4/model.ckpt-3223090',
    trust_repo=True,
) # creates 4 channels; use output_sources8 to separate into 8 channels

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
import torch
m = torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'YAMNet',trust_repo=True)
m.predict(['test.wav']) # returns dataframe of per-class scores
m.generate_embeddings(['test.wav']) # returns dataframe of embeddings
```


### rana_sierrae_cnn: 

Detect underwater vocalizations of _Rana sierrae_, the Sierra Nevada Yellow-legged Frog

example: 
```python
import torch
m = torch.hub.load('kitzeslab/bioacoustics-model-zoo', 'rana_sierrae_cnn',trust_repo=True)
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


## Troubleshooting 

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
