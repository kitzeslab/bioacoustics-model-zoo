# %%
from pathlib import Path
import sys

sys.path.append(str(Path("..").resolve()))
sys.path.append(str(Path(".").resolve()))


from bioacoustics_model_zoo import perch

# %%
model_dir = "/Users/SML161/bioacoustics-model-zoo/resources/perch_0.1.2/"
m = perch.Perch(model_dir)

# %%

# %%

files = ["/Users/SML161/sample_audio/1min_ampr.mp3"]


# %%
emb = m.generate_embeddings(files, batch_size=1)
# %%
emb2 = m.generate_embeddings(files, batch_size=32)

# %%
