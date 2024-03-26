from typing import List, Dict
import numpy as np
from abc import ABC, abstractmethod

# Abstrat class for encoder, has encode as abstract method
class TextEncoder(ABC):
    @abstractmethod
    def encode(self, embeddings: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        pass