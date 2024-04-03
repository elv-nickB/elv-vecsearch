import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import pickle as pkl

from src.embedding.abstract import TextEncoder

# Args:
#   x: np.ndarray of shape (N, d) where N is the number of embeddings and d is the dimension of the embeddings
#   K: number of clusters to reduce to
#   T: threshold for removing similar embeddings
# Returns:
#   np.ndarray of shape (K, d) where K is the number of clusters
def tag_clean(x: np.ndarray, K: int, T: float) -> np.ndarray:
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

def load_encoder(model_name: str) -> TextEncoder:
    return SentenceTransformer(model_name)

# Args:
#   model_name: str, name of the model to load
#   cache_path (optional): str, path to cache file (pkl format)
# Returns:
#   TextEncoder instance with caching. Initializes the cache from the provided path 
#   if cache_path is not None, else initializes with an empty cache.
def load_encoder_with_cache(model_name: str, cache_path: str=None) -> TextEncoder:
    class CacheEncoder(TextEncoder):
        def __init__(self, model_name: str, cache: dict):
            self.encoder = SentenceTransformer(model_name)
            self.cache = cache
        def encode(self, text: str) -> np.ndarray:
            if text not in self.cache:
                self.cache[text] = self.encoder.encode(text, show_progress_bar=False).squeeze()
            return self.cache[text]
    return CacheEncoder(model_name, {}) if cache_path is None else CacheEncoder(model_name, pkl.load(open(cache_path, 'rb')))