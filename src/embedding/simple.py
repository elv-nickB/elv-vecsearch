from typing import Dict, List
import numpy as np

from src.embedding.abstract import TextEncoder, DocEncoder

class SimpleEncoder(DocEncoder):
    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder

    def encode(self, embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():
            res[fname] = self.encoder.encode(text)
        return res