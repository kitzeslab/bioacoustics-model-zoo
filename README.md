# bioacoustics-model-zoo
Pre-trained models for bioacoustic classification tasks

# Basic usage

### List: 
list available models in the GitHub repo [bioacoustics-model-zoo](https://github.com/kitzeslab/bioacoustics-model-zoo/)
```
import torch
torch.hub.list('kitzeslab/bioacoustics-model-zoo')
```

### Load: 
Get a ready-to-use model object: choose from the models listed in the previous command
```
model = torch.hub.load('kitzeslab/bioacoustics-model-zoo','rana_sierrae_cnn')
```

### Inference:

`model` is an OpenSoundscape CNN object (or other class) which you can use as normal. 

For instance, use the model to generate predictions on an audio file: 

```
audio_file_path = './hydrophone_10s.wav'
scores = model.predict([audio_file_path],activation_layer='softmax')
scores
```

# Model list

`rana_sierrae_cnn`: detect underwater vocalizations of Rana sierrae 

``

## Troublshooting 

Installing TensorFlow can be tricky, and it may not be possible to have cuda-enabled tensorflow in the same environment as pytorch. In this case, you can install a cpu-only version of tensorflow (`pip install tensorflow-cpu`). You may want to start with a fresh environment, or uninstall tensorflow and nvidia-cudnn-cu11 then reinstall pytorch with the appropriate nvidia-cudnn-cu11, to avoid having the wrong cudnn for PyTorch. 

Installing tensorflow: Carefully follow the [directions](https://www.tensorflow.org/install/pip) for your system. Note that models provided in this repo might require the specific nvidia-cudnn-cu11 version 8.6.0, which could conflict with the version required for pytorch. 