from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod

# Abstract class for document encoder
class DocEncoder(ABC):
    # Args:
    #   document: Dict[str, List[str]], dictionary of embeddings from field-name -> list of field instances
    # Returns:
    #   Dict[str, np.ndarray], dictionary of embeddings from field -> np.ndarray of embeddings
    @abstractmethod
    def encode(self, document: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        pass

# Abstract class for text encoder
class TextEncoder(ABC):
    # Args:
    #   text: str, text to encode
    # Returns:
    #   1-D np.ndarray, encoded text
    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass