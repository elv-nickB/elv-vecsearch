
from format import *
import os
from typing import List, Dict
from elv_client_py import ElvClient
import logging
from typing import List, Dict, Tuple
from functools import reduce

from src.index import Index
from src.utils import timeit
from src.rank import Ranker
from src.query_understanding import QueryProcessor
from src.index import Index
from src.rank import Ranker
from src.format import SearchArgs, SearchOutput

"""
This module is responsible for tying all the conceptual blocks in the design together to form a complete search pipeline.

The main entrypoint is the Search class and search function.
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Searcher():
    def __init__(self, index_qid: str, client: ElvClient, processor: QueryProcessor, index: Index, ranker: Ranker):
        self.client = client
        self.processor = processor
        self.index = index
        self.ranker = ranker
        self.index_qid = index_qid

    def search(self, args: SearchArgs) -> SearchOutput:
        with timeit("Embedding query and processing it"):
            query = self.processor.process_query(self.index_qid, args)
        with timeit("Retrieving document uids from vector index"):
            uids_per_field = [self.index.search(query.embedding, field, k=args['max_total']) for field in args['search_fields']]
            # concatenate the results from different fields and deduplicate
            uids = set(reduce(lambda x, y: x+y, uids_per_field))
        ranked_uids = self.ranker.rank(query, uids, args['max_total'])
        # for retrieving the original order of the results after they are shuffled in the next step
        pos_map = self._get_pos_map(ranked_uids)
        # delegate search formatting and additional features to fabric search engine
        with timeit("Searching uids on fabric"):
            del args['terms']
            res = self.fabric_search([x[0] for x in ranked_uids], args)
        res = SearchOutput().load(res)
        # fabric search won't preserve original order, so we will re-sort the results
        with timeit("re-sorting the results"):
            res['results'].sort(key=lambda x: pos_map[self._uid_from_result(x)])
        # add the scores from the ranker to the results
        for rr, sr in zip(ranked_uids, res['results']):
            sr['score'] = rr[1]
        return res

    def fabric_search(self, uids: List[str], args: SearchArgs) -> SearchOutput:
        # Select the documents from index based on uids
        query = ' '.join(f'uid:\"{uid}\"' for uid in uids)
        if "filters" in args and args["filters"] != "":
            query = f"({query}) AND ({args['filters']})"
        logging.info(f"Querying fabric with {query}")
        args["filters"] = query
        return self.client.search(object_id=self.index_qid, query=args)

    def _get_pos_map(self, l: List[Tuple[str, float]]) -> Dict[str, int]:
        return {l[i][0]: i for i in range(len(l))}

    def _uid_from_result(self, res: Dict[str, str]) -> str:
        return f"{res['hash']}{res['prefix']}"
