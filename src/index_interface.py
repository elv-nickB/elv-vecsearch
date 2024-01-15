from abc import ABC, abstractmethod
from typing import List, Dict

class Index(ABC):
    @abstractmethod
    def add(self, field: str, uid: str, text: List[str]) -> None:
        pass

    @abstractmethod
    def search(self, fields: List[str], query: str) -> List[str]:
        pass

    @abstractmethod
    def get_repr(self, uid: str) -> Dict[str, List[str]]:
        pass