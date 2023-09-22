# %%
from pathlib import Path
import sys

sys.path.append(str(Path("..").resolve()))
sys.path.append(str(Path(".").resolve()))


from bioacoustics_model_zoo import perch

# %%
m = perch.Perch()

# %%

files = ["/Users/SML161/sample_audio/20min_audiomoth.mp3"]


# %%
emb = m.generate_embeddings(files, batch_size=1)
# %%
emb2 = m.generate_embeddings(files, batch_size=128)

# %%
