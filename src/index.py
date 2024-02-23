from abc import ABC, abstractmethod
from typing import List, Dict, Iterable, Callable
import numpy as np
import os
import dill
import shutil
import faiss

# field name -> list of embeddings
VectorDocument = Dict[str, List[np.ndarray]]

class Index(ABC):
    """Abstract class represening the Index block in the design. This class is meant to be subclassed by a specific vector index implementation."""

    # Args:
    #  uid: unique identifier for the document
    #  field: field to add the embeddings to
    #  embeddings: embeddings to add to the index
    # Side Effects:
    #  After committing, Searching for an embedding in the index under the the given field can return the given uid.
    @abstractmethod
    def add(self, uid: str, field: str, embeddings: Iterable[np.ndarray]) -> None:
        pass

    # Args:
    #  field: field to search in
    #  query: query embedding (1 dimensional np.ndarray)
    # Returns:
    #  List of uids of documents that have embeddings close to the query
    @abstractmethod
    def search(self, query: np.ndarray, field: str, k: int) -> List[str]:
        pass

    # Returns:
    #  List of searchable fields in the index. 
    @abstractmethod
    def get_fields(self) -> List[str]:
        pass
    
    # Args:
    #  uid: unique identifier for the document
    # Returns:
    #  Dictionary mapping field names to a list of the original embeddings added to the index under that field for the given uid
    @abstractmethod
    def get_embeddings(self, uid: str) -> VectorDocument:
        pass

    # Side Effects:
    #  Make the index searchable
    @abstractmethod
    def commit(self) -> None:
        pass

    # Side Effects:
    #  Changes the location of the index on disk
    @abstractmethod
    def set_path(self, path: str) -> None:
        pass

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
        os.makedirs(os.path.join(self.path, 'vector_data'), exist_ok=True)

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
    def add(self, field: str, uid: str, embeddings: List[np.ndarray]) -> None:
        if field not in self.id_map:
            self.id_map[field] = []
            self.indices[field] = self.get_index()
        self.indices[field].add(embeddings)
        for _ in embeddings:
            self.id_map[field].append(uid)
        if not os.path.exists(os.path.join(self.path, 'vector_data', self.uid_to_path(uid))):
            os.makedirs(os.path.join(self.path, 'vector_data', self.uid_to_path(uid)))
        with open(os.path.join(self.path, 'vector_data', self.uid_to_path(uid), f'{field}.pkl'), 'wb') as f:
            np.save(f, embeddings)

    """
    Args:
        query: Query embedding to search with
        field: field to search on. 
        k: Number of results from each field to retrieve in retrieval step
    Returns:
        List of the top k uids
    """
    def search(self, query: np.ndarray, field: str, k: int=500) -> List[str]:
        assert field in self.indices, f"Index is missing some of the requested field: searching={field}, available={self.get_fields()}"
        ids = self.indices[field].search(np.expand_dims(query, 0), k)[1]
        ids = ids.squeeze(0)
        uids = [self.id_map[field][i] for i in filter(lambda x: x >= 0, ids)]
        return uids
    
    def get_fields(self) -> List[str]:
        return list(self.indices.keys())

    # Retrieves all vector data from disk 
    def get_embeddings(self, uid: str) -> Dict[str, List[np.ndarray]]:
        path = os.path.join(self.path, 'vector_data', self.uid_to_path(uid))
        if not os.path.exists(path):
            raise ValueError(f"Vector data for {uid} does not exist")
        data = {}
        for field_data in os.listdir(path):
            field = field_data[0:-4]
            with open(os.path.join(path, field_data), 'rb') as f:
                data[field] = np.load(f)
        return data
    
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

    def uid_to_path(self, uid: str) -> str:
        return uid.replace('/', '_')