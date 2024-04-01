
from typing import List, Dict
from elv_client_py import ElvClient
import logging
from typing import List, Dict, Tuple
from functools import reduce

from src.index.faiss import Index
from src.utils import timeit
from src.ranking.simple import SimpleRanker
from src.query_processing.simple import SimpleProcessor
from src.index.faiss import Index
from src.format import SearchArgs, SearchOutput
from src.search.abstract import Searcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleSearcher(Searcher):
    def __init__(self, index_qid: str, client: ElvClient, processor: SimpleProcessor, index: Index, ranker: SimpleRanker):
        self.client = client
        self.processor = processor
        self.index = index
        self.ranker = ranker
        self.index_qid = index_qid

    def search(self, args: SearchArgs) -> SearchOutput:
        with timeit("Embedding query and processing it"):
            query = self.processor.process_query(self.index_qid, args)
        if args["uids"]:
            # then we are skipping the retrieval step and just ranking the provided uids
            uids = args["uids"]
        else:
            # retrieve the uids from the index
            with timeit("Retrieving document uids from vector index"):
                uids_per_field = [self.index.search(query["embedding"], field, k=100) for field in args['search_fields']]
                # concatenate the results from different fields and deduplicate
                uids = set(reduce(lambda x, y: x+y, uids_per_field))
                # Also search the top results from the fabric search and perform ranking on these. 
                fs_args = {
                    "terms": query["corrected_query"],
                    "semantic": True,
                    "max_total": 100,
                    "limit": 100,
                    "display_fields": ["uid"],
                }
                r = self.client.search(object_id=self.index_qid, query=fs_args)
                uids.update(x["fields"]["uid"][0] for x in r["results"])
        with timeit("Ranking documents"):
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
        if "debug" in args:
            res['debug'] = {}
            if "weights" in query:
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
