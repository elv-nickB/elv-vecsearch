from typing import List, Dict, Callable
import numpy as np
import os
import dill
import shutil
import faiss
import logging
import h5py

from src.index.abstract import Index

class FaissIndex(Index):
    # subclass used to store metadata about the index for easy serialization
    class Meta:
        # call to initialize a new faiss index 
        IndexConstructor = Callable[[], faiss.Index]
        def __init__(self, path: str, index_builder: IndexConstructor, id_map: Dict[str, List[str]]) -> None:
            self.path = path
            self.index_builder = index_builder
            self.id_map = id_map

    def __init__(self, path: str, index_builder: Meta.IndexConstructor) -> None:
        os.makedirs(path, exist_ok=True)
        self.path = path
        # maps field -> faiss index
        self.indices = {}
        # maps field -> list of uids (allows us to retrieve uid from index after search)
        self.id_map = {}
        self.get_index = index_builder
        self.vector_data = h5py.File(os.path.join(self.path, 'vector_data.h5'), 'a')

    @staticmethod
    def from_path(path: str) -> Index:
        indices = {}
        meta = None
        for fname in os.listdir(path):
            if fname == 'meta.pkl':
                with open(os.path.join(path, fname), 'rb') as f:
                    meta = dill.load(f)
            elif fname.endswith('.index'):
                field, _ = os.path.splitext(fname)
                index = faiss.read_index(os.path.join(path, fname))
                indices[field] = index
        if meta is None:
            raise ValueError("Index directory does not contain metadata file meta.pkl")
        index.vector_data = h5py.File(os.path.join(path, 'vector_data.h5'), 'a')
        index = FaissIndex(meta.path, meta.index_builder)
        index.indices = indices
        index.id_map = meta.id_map
        return index
    
    """
    Args:
        field: Field under which to add the text
        uid: Unique identifier of the document
        text: Text to add to the index
    Side Effects:
        Adds the text to the index so that it can be searched
    """
    def add(self, field: str, uid: str, embeddings: np.ndarray) -> None:
        if field not in self.id_map:
            self.id_map[field] = []
            self.indices[field] = self.get_index()
        self.indices[field].add(embeddings)
        for _ in embeddings:
            self.id_map[field].append(uid)
        self.vector_data[f'{uid}/{field}'] = embeddings

    """
    Args:
        query: Query embedding to search with
        field: field to search on. 
        k: Number of results from each field to retrieve in retrieval step
    Returns:
        List of the top k uids
    """
    def search(self, query: np.ndarray, field: str, k: int=500) -> List[str]:
        if field not in self.indices:
            logging.error(f"Index is missing some of the requested field: searching={field}, available={self.get_fields()}")
            return []
        ids = self.indices[field].search(np.expand_dims(query, 0), k)[1]
        ids = ids.squeeze(0)
        uids = [self.id_map[field][i] for i in filter(lambda x: x >= 0, ids)]
        return uids
    
    def get_fields(self) -> List[str]:
        return list(self.indices.keys())

    # Retrieves all vector data from disk 
    def get_embeddings(self, uid: str) -> Dict[str, np.ndarray]:
        return {field: np.array(self.vector_data[f'{uid}'][field]) for field in self.get_fields() if f'{uid}/{field}' in self.vector_data}
    
    # Backs up index to disk. 
    def commit(self) -> None:
        for field, index in self.indices.items():
            faiss.write_index(index, os.path.join(self.path, f'{field}.index'))
        self._save_meta()

    def set_path(self, new_path: str) -> None:
        shutil.move(self.path, new_path)
        self.path = new_path
        # update the path in the meta file
        self._save_meta()

    def _save_meta(self) -> None:
        meta = FaissIndex.Meta(self.path, self.get_index, self.id_map)
        with open(os.path.join(self.path, 'meta.pkl'), 'wb') as f:
            dill.dump(meta, f)