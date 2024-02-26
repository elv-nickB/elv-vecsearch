from typing import Callable, List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# NOTE: we can make an abstract class for the Embedder block as well, but it's not necessary for now

# function taking fieldname, list of embeddings -> embeddings
TextEncoder = Callable[[Dict[str, List[str]]], np.ndarray]

categorical_tracks = ["f_celebrity", "f_action", "f_segment", "f_logo", "f_landmark"]

def postprocess(x: np.ndarray, K: int, T: float) -> np.ndarray:
    to_remove = []
    for i in range(x.shape[0]):
        for j in range(i+1, x.shape[0]):
            if np.dot(x[i], x[j]) > T:
                to_remove.append(j)
    x = np.delete(x, to_remove, axis=0)
    if x.shape[0] > K:
        kmeans = KMeans(n_clusters=K, n_init=10)
        kmeans.fit(x)
        x = kmeans.cluster_centers_
    return x

# TODO: if we need to optimize we can batch embed all of them at once
def get_encoder_with_cache(model_name: str, K: int=None, T: float=None) -> TextEncoder:
    model = SentenceTransformer(model_name)
    cache = {}
    def encode(embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        res = {}
        for fname, text in embeddings.items():    
            if fname in categorical_tracks:
                text = [", ".join(text)]
            for t in text:
                if t not in cache:
                    cache[t] = model.encode(t).squeeze()
            e = np.array([cache[t] for t in text])
            if K or T:
                assert K and T, "K and T must be set together"
                res[fname] = postprocess(e, K, T)
        return res
    return encode
