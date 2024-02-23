from typing import Callable, List
import numpy as np
from sentence_transformers import SentenceTransformer

# NOTE: we can make an abstract class for the Embedder block as well, but it's not necessary for now

# function taking (text) -> embedding
TextEncoder = Callable[[str], np.ndarray]

def get_encoder(model_name: str) -> TextEncoder:
    model = SentenceTransformer(model_name)
    return lambda text: model.encode(text, show_progress_bar=False)

def get_encoder_with_cache(model_name: str) -> TextEncoder:
    model = SentenceTransformer(model_name)
    cache = {}
    def encode(text: List[str]) -> np.ndarray:
        for t in text:
            if t not in cache:
                cache[t] = model.encode(t, show_progress_bar=False)
        return np.array([cache[t] for t in text])
    return encode