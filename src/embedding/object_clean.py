from typing import List, Dict
import numpy as np

from src.vars import *
from src.embedding.abstract import TextEncoder, DocEncoder
from src.embedding.utils import tag_clean 

# Encoder for video tags
# Performs git cleanup and also merges categorical tracks into a single embedding
class ObjectCleanEncoder(DocEncoder):
    def __init__(self, encoder: TextEncoder, K: int=None, T: float=None):
        self.encoder = encoder
        self.K = K
        self.T = T

    # Args:
    #   embeddings: Dict of embeddings with keys as the field names and list of text values as values
    #   K: number of clusters to reduce to
    #   T: threshold for removing similar embeddings
    # Note: K and T are optional, omitting them will keep all embeddings
    # Returns:
    #   Dict of embeddings with keys as the field names and values as the embeddings
    def encode(self, embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():    
            if fname in CAT_TRACKS:
                text = [", ".join(text)]
            e = np.array([self.encoder.encode(t) for t in text])
            if fname == "f_object" or fname == "f_characters":
                e = tag_clean(e, self.K, self.T)
            res[fname] = e
        return res    
    
    def set_t_k(self, T: float, K: int) -> None:
        self.T = T
        self.K = K