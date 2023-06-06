dependencies = ['torch', 'opensoundscape']
import torch
from opensoundscape import CNN

## see instructions here: 
## https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url

# BUILTIN_MODELS = {}

# def register_model(name: Optional[str] = None) -> Callable[[Callable[..., M]], Callable[..., M]]:
#     def wrapper(fn: Callable[..., M]) -> Callable[..., M]:
#         key = name if name is not None else fn.__name__
#         if key in BUILTIN_MODELS:
#             raise ValueError(f"An entry is already registered under the name '{key}'.")
#         BUILTIN_MODELS[key] = fn
#         return fn

#     return wrapper
  
# @register_model
def rana_sierrae_cnn(pretrained=True):
  """Load CNN that detects Rana Sierrae vocalizations"""
  model = CNN(# add params to configure)
  
  #make any necessary adjustments here
  
  model.load_state_dict(torch.hub.load_state_dict_from_url('a/public/url.pth', progress=False))
  
  return model
