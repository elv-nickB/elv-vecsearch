from abc import ABC, abstractmethod
from src.format import SearchArgs, SearchOutput, ClipSearchOutput

class Searcher(ABC):
    @abstractmethod
    def search(self, args: SearchArgs) -> SearchOutput | ClipSearchOutput:
        pass