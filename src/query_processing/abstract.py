
from abc import ABC, abstractmethod
from typing import Any

from src.format import SearchArgs

Query = Any

# Abstract class representing Query Understanding block
class QueryProcessor(ABC):
    # Args:
    #   content_id: id of the index object
    #   query: user query
    #  Returns:
    #   A processed query, the type of which is defined by the subclass. Will be passed to core search block.
    @abstractmethod
    def process_query(self, content_id: str, query: SearchArgs) -> Query:
        pass