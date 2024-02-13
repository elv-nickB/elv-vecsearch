
from format import *
from abc import ABC, abstractmethod
from typing import List, Dict
from elv_client_py import ElvClient
import logging
from typing import List, Dict, Tuple
from functools import reduce

from src.index import Index
from src.utils import timeit
from src.rank import SimpleRanker
from src.query_understanding import SimpleQueryProcessor
from src.index import Index
from src.format import SearchArgs, SearchOutput

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Searcher(ABC):
    @abstractmethod
    def search(self, args: SearchArgs) -> SearchOutput:
        pass

class SimpleSearcher(Searcher):
    def __init__(self, index_qid: str, client: ElvClient, processor: SimpleQueryProcessor, index: Index, ranker: SimpleRanker):
        self.client = client
        self.processor = processor
        self.index = index
        self.ranker = ranker
        self.index_qid = index_qid

    def search(self, args: SearchArgs) -> SearchOutput:
        with timeit("Embedding query and processing it"):
            query = self.processor.process_query(self.index_qid, args)
        with timeit("Retrieving document uids from vector index"):
            uids_per_field = [self.index.search(query["embedding"], field, k=args['max_total']) for field in args['search_fields']]
            # concatenate the results from different fields and deduplicate
            uids = set(reduce(lambda x, y: x+y, uids_per_field))
        ranked_uids = self.ranker.rank(uids, args['max_total'], query)
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
        if args["debug"]:
            res['debug'] = {"query weights": query["weights"]}
        return res

    # NOTE: if we want to build a new searcher we can move this method out of this class so others can call it. 
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
