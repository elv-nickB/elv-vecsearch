from elv_client_py import ElvClient
from typing import Dict, Any
from abc import ABC, abstractmethod
from marshmallow import Schema, fields as field

from src.embedding import TextEncoder
from src.format import SearchArgs, Any

FieldWeights = Dict[str, float]

# Abstract class representing Query Understanding block
class QueryProcessor(ABC):
    # Args:
    #   content_id: id of the index object
    #   query: user query
    #  Returns:
    #   A processed query, the type of which is defined by the subclass. Will be passed to core search block.
    @abstractmethod
    def process_query(self, content_id: str, query: SearchArgs) -> Any:
        pass

class SimpleQueryProcessor(QueryProcessor):

    # Schema for the processed query:
    # NOTE: Query processors will return arbitrary dictionary-like output, so we can accomodate a variety of use cases.
    # But they may define the schema like this, so developers can at least understand the expected output. 
    class ProcessedQuery(Schema):
        content_id = field.Str()
        args = field.Nested(SearchArgs)
        corrected_query = field.Str()
        weights = field.Dict(field.Str(), field.Float())
        embedding = Any(allow_none=True)

    # Args:
    #   client: ElvClient instance for fabric interaction
    #   content_id: ID of index object to retrieve term weights from
    #   encoder: TextEncoder instance for encoding queries
    def __init__(self, client: ElvClient, encoder: TextEncoder):
        self.client = client
        self.encoder = encoder

    def process_query(self, content_id: str, query: SearchArgs) -> ProcessedQuery:
        corrected_query = self._correct_query(query["terms"])
        weights = self._get_weights_from_query(content_id, corrected_query)
        embedding = self.encoder(corrected_query)
        res = {"content_id": content_id, "corrected_query": corrected_query, "weights": weights, "embedding": embedding}
        res = self.ProcessedQuery().load(res)
        res["args"] = query
        return res
    
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
        # normalize weights by field
        total = sum(wt.values())
        for field in wt:
            wt[field] /= total
        return wt

    def _correct_query(self, query: str) -> str:
        return query
    