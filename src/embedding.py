from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pickle as pkl

# Abstrat class for encoder, has encode as abstract method
class TextEncoder(ABC):
    @abstractmethod
    def encode(self, embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        pass

class SimpleEncoder(TextEncoder):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def encode(self, embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():
            res[fname] = self.model.encode(text, show_progress_bar=False)
        return res
    
# Encoder for video tags
# Performs git cleanup and also merges categorical tracks into a single embedding
class VideoTagEncoder(TextEncoder):
    categorical_tracks = ["f_celebrity", "f_action", "f_segment", "f_logo", "f_landmark"]
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.cache = {}

    # Args:
    #   embeddings: Dict of embeddings with keys as the field names and list of text values as values
    #   K: number of clusters to reduce to
    #   T: threshold for removing similar embeddings
    # Note: K and T are optional, omitting them will keep all embeddings
    # Returns:
    #   Dict of embeddings with keys as the field names and values as the embeddings
    def encode(self, embeddings: Dict[str, List[str]], K: int=None, T: float=None) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():    
            if fname in VideoTagEncoder.categorical_tracks:
                text = [", ".join(text)]
            for t in text:
                if t not in self.cache:
                    self.cache[t] = self.model.encode(t, show_progress_bar=False).squeeze()
            e = np.array([self.cache[t] for t in text])
            if fname == "f_object":
                e = _tag_clean(e, K, T)
            res[fname] = e
        return res    
    
    def save_cache(self, path: str) -> None:
        with open(path, 'wb') as f:
            pkl.dump(self.cache, f)

    def load_cache(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.cache = pkl.load(f)
    
    def set_t_k(self, T: float, K: int) -> None:
        self.T = T
        self.K = K

# Args:
#   x: np.ndarray of shape (N, d) where N is the number of embeddings and d is the dimension of the embeddings
#   K: number of clusters to reduce to
#   T: threshold for removing similar embeddings
# Returns:
#   np.ndarray of shape (K, d) where K is the number of clusters
def _tag_clean(x: np.ndarray, K: int, T: float) -> np.ndarray:
    to_remove = []
    if T:
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                if np.dot(x[i], x[j]) > T:
                    to_remove.append(j)
    x = np.delete(x, to_remove, axis=0)
    if K:
        if x.shape[0] > K:
            kmeans = KMeans(n_clusters=K, n_init=10)
            kmeans.fit(x)
            x = kmeans.cluster_centers_
    return x