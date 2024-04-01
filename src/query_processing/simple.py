
from typing import Any
from marshmallow import Schema, fields as field
from elv_client_py import ElvClient

from src.embedding.abstract import TextEncoder
from src.format import SearchArgs, Any
from src.query_processing.abstract import QueryProcessor

# This processor includes word weights in the processed query output.
class SimpleProcessor(QueryProcessor):

    # Schema for the processed query:
    # NOTE: Query processors will return arbitrary dictionary-like output, so we can accomodate a variety of use cases.
    # But they may define the schema like this, so developers can at least understand the expected output. 
    class ProcessedQuery(Schema):
        content_id = field.Str()
        args = field.Nested(SearchArgs)
        corrected_query = field.Str()
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
        embedding = self.encoder.encode(query["terms"])
        res = {"content_id": content_id, "corrected_query": corrected_query, "embedding": embedding}
        res = self.ProcessedQuery().load(res)
        res["args"] = query
        return res

    def _correct_query(self, query: str) -> str:
        return query.lower()