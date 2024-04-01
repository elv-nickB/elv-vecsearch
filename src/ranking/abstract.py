
from typing import List, Tuple, Iterable, Any
from abc import ABC, abstractmethod
from typing import Callable
from src.index.abstract import VectorDocument
from src.query_processing.abstract import Query

# function taking (Query, VectorDocument) -> score
Scorer = Callable[[Query, VectorDocument], float] 

class Ranker(ABC):
    """
    Abstract class representing the Re-ranking block in the design. This class is meant to be subclassed by a specific ranking implementation. 
    The rank method should take a list of uids and a query and return a list of uids and their associated scores (in sorted order).

    The query format is undefined here, the specific implementation will define the expected input (see SimpleRanker for an example).
    """
    @abstractmethod
    def rank(self, uids: Iterable[str], limit: int, query: Any) -> List[Tuple[str, float]]:
        pass