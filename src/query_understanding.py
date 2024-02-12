from elv_client_py import ElvClient
from typing import Dict
from abc import ABC, abstractmethod
import numpy as np

from src.embedding import TextEncoder
from src.format import SearchArgs

FieldWeights = Dict[str, float]

# Represents all data that should be passed to the Search block. For now, this is just a query and its weights
class Query:
    def __init__(self, content_id: str, args: SearchArgs, corrected_query: str, weights: FieldWeights, embedding: np.ndarray):
        self.content_id = content_id
        self.args = args
        self.corrected_query = corrected_query
        self.weights = weights
        self.embedding = embedding

# Abstract class representing Query Understanding block
class QueryProcessor(ABC):
    # Args:
    #   content_id: id of the index object
    #   query: user query
    #  Returns:
    #   Query object representing the query, to be passed to search block
    @abstractmethod
    def process_query(self, content_id: str, query: SearchArgs) -> Query:
        pass

class SimpleQueryProcessor(QueryProcessor):
    # Args:
    #   client: ElvClient instance for fabric interaction
    #   content_id: ID of index object to retrieve term weights from
    #   encoder: TextEncoder instance for encoding queries
    def __init__(self, client: ElvClient, encoder: TextEncoder):
        self.client = client
        self.encoder = encoder

    def process_query(self, content_id: str, query: SearchArgs) -> Query:
        corrected_query = self._correct_query(query["terms"])
        weights = self._get_weights_from_query(content_id, corrected_query)
        embedding = self.encoder(corrected_query)
        return Query(content_id, query, corrected_query, weights, embedding)
    
    # Args:
    #   content_id: ID of index object to retrieve term weights from
    #   query: text query
    #
    # Returns:
    #   Dict[str, float] mapping from field to weight for the given query
    def _get_weights_from_query(self, content_id: str, query: str) -> FieldWeights:
        terms = query.split(' ')
        term_weights = self.client.content_object_metadata(object_id=content_id, metadata_subtree='search/weights', select=terms)
        fields = list(next(iter(term_weights.values())).keys())
        wt = {f: 0 for f in fields}
        for term in term_weights:
            if term not in term_weights:
                continue
            for field in wt:
                wt[field] += term_weights[term][field]
        return wt

    def _correct_query(self, query: str) -> str:
        return query