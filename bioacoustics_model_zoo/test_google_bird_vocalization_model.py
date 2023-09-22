import sys
from pathlib import Path

sys.path.append(str(Path("../").resolve()))


from google_bird_vocalization_classifier import (
    google_bird_vocalization_classifier,
)
import numpy as np

m = google_bird_vocalization_classifier(
    "https://tfhub.dev/google/bird-vocalization-classifier/2"  # should be 3, but 3 has bug
)
emb = m.generate_embeddings(["/Users/SML161/a.mp3"])
logits = m.generate_logits(["/Users/SML161/a.mp3"])
preds = m.predict(["/Users/SML161/a.mp3"])

# which classes does the model think are present?
[c for c in classes if preds[c].max() > 0]
