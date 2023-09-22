import sys
from pathlib import Path

sys.path.append(str(Path("../").resolve()))
# sys.path.append(str(Path("./").resolve()))

from bioacoustics_model_zoo.google_bird_vocalization_classifier import (
    google_bird_vocalization_classifier,
)
import numpy as np

m = google_bird_vocalization_classifier(
    "https://tfhub.dev/google/bird-vocalization-classifier/2"  # should be 3, but 3 has bug
)

f = "/Users/SML161/sample_audio/birds_10s.wav"
emb = m.generate_embeddings([f])
logits = m.generate_logits(["/Users/SML161/a.mp3"])
preds = m.predict(["/Users/SML161/a.mp3"])

# which classes does the model think are present?
[c for c in m.classes if preds[c].max() > 0]
