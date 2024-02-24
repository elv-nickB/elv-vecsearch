from typing import Callable, List
import numpy as np
from sentence_transformers import SentenceTransformer

# NOTE: we can make an abstract class for the Embedder block as well, but it's not necessary for now

# function taking fieldname, list of embeddings -> embeddings
TextEncoder = Callable[[str, List[str]], np.ndarray]

categorical_tracks = ["f_celebrity", "f_action", "f_segment", "f_logo", "f_landmark"]

def get_encoder_with_cache(model_name: str) -> TextEncoder:
    model = SentenceTransformer(model_name)
    cache = {}
    def encode(fname: str, text: List[str]) -> np.ndarray:
        if fname in categorical_tracks:
            text = [", ".join(text)]
        for t in text:
            if t not in cache:
                cache[t] = model.encode(t).squeeze()
        return np.array([cache[t] for t in text])
    return encode