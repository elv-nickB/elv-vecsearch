from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding.abstract import TextEncoder

class SimpleEncoder(TextEncoder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():
            res[fname] = self.model.encode(text, show_progress_bar=False)
        return res