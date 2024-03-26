from abc import ABC, abstractmethod
from src.format import SearchArgs, SearchOutput

class Searcher(ABC):
    @abstractmethod
    def search(self, args: SearchArgs) -> SearchOutput:
        pass