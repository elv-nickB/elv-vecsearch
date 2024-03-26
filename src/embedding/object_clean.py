from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pickle as pkl

from src.embedding.abstract import TextEncoder

# Encoder for video tags
# Performs git cleanup and also merges categorical tracks into a single embedding
class VideoTagEncoder(TextEncoder):
    categorical_tracks = ["f_celebrity", "f_action", "f_segment", "f_logo", "f_landmark"]
    def __init__(self, model_name: str, K: int=None, T: float=None):
        self.model = SentenceTransformer(model_name)
        self.cache = {}
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
            if fname in VideoTagEncoder.categorical_tracks:
                text = [", ".join(text)]
            for t in text:
                if t not in self.cache:
                    self.cache[t] = self.model.encode(t, show_progress_bar=False).squeeze()
            e = np.array([self.cache[t] for t in text])
            if fname == "f_object" or fname == "f_characters":
                e = _tag_clean(e, self.K, self.T)
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
    if T:
        to_remove = []
        for i in range(x.shape[0]):
            for j in range(i+1, x.shape[0]):
                if np.dot(x[i], x[j]) > T:
                    to_remove.append(j)
        x = np.delete(x, to_remove, axis=0)
    if K:
        if K == 1:
            x = np.mean(x, axis=0).reshape(1, -1)
        elif x.shape[0] > K:
            kmeans = KMeans(n_clusters=K, n_init=10)
            kmeans.fit(x)
            x = kmeans.cluster_centers_
    return x