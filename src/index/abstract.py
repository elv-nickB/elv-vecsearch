
from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

# field name -> list of embeddings
VectorDocument = Dict[str, List[np.ndarray]]

class Index(ABC):
    """Abstract class represening the Index block in the design. This class is meant to be subclassed by a specific vector index implementation."""

    # Args:
    #  uid: unique identifier for the document
    #  field: field to add the embeddings to
    #  embeddings: embeddings to add to the index (2D np.ndarray, each row is an embedding to add to the index)
    # Side Effects:
    #  After committing, Searching for an embedding in the index under the the given field can return the given uid.
    @abstractmethod
    def add(self, uid: str, field: str, embeddings: np.ndarray) -> None:
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