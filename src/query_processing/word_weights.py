
import logging
from typing import Dict, Any
from marshmallow import Schema, fields as field
from elv_client_py import ElvClient

from src.embedding.abstract import DocEncoder
from src.format import SearchArgs, Any
from src.query_processing.simple import SimpleProcessor

FieldWeights = Dict[str, float]

# This processor includes word weights in the processed query output.
class WordWeightsProcessor(SimpleProcessor):

    # Includes weights in the processed query output
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
    def __init__(self, client: ElvClient, encoder: DocEncoder):
        super().__init__(client, encoder)

    def process_query(self, content_id: str, query: SearchArgs) -> ProcessedQuery:
        res = super().process_query(content_id, query)
        res['weights'] = self._get_weights_from_query(content_id, res["corrected_query"])
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
        # Hack if term_weights is None
        if term_weights is None:
            term_weights = self.client.content_object_metadata(object_id=content_id, metadata_subtree='search/weights', select=["a", "the"])
            logging.error(f"Term weights were not found for the given query {query}. Using default weights.")
            assert term_weights is not None
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