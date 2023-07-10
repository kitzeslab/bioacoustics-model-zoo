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

`model` is an OpenSoundscape CNN object which you can use as normal. 

For instance, use the model to generate predictions on an audio file: 

```
audio_file_path = './hydrophone_10s.wav'
scores = model.predict([audio_file_path],activation_layer='softmax')
scores
```
