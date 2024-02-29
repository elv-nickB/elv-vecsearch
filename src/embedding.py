from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

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
            res[fname] = self.model.encode(text)
        return res
    
# Encoder for video tags
# Performs git cleanup 
class VideoTagEncoder():
    categorical_tracks = ["f_celebrity", "f_action", "f_segment", "f_logo", "f_landmark"]
    def __init__(self, model_name: str, K: int=None, T: float=None):
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        self.K = K
        self.T = T

    def encode(self, embeddings: Dict[str, List[str]], K: int=None, T: float=None) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():    
            if fname in VideoTagEncoder.categorical_tracks:
                text = [", ".join(text)]
            for t in text:
                if t not in self.cache:
                    self.cache[t] = self.model.encode(t).squeeze()
            e = np.array([self.cache[t] for t in text])
            if fname == "f_object":
                e = _tag_clean(e, K, T)
            res[fname] = e
        return res    

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