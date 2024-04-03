
from typing import List, Dict
from elv_client_py import ElvClient
import logging
from typing import List, Dict, Any
from functools import reduce

from src.index.faiss import Index
from src.utils import timeit
from src.ranking.simple import SimpleRanker
from src.query_processing.simple import SimpleProcessor
from src.index.faiss import Index
from src.format import SearchArgs, SearchOutput, ClipSearchOutput
from src.search.abstract import Searcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleSearcher(Searcher):
    def __init__(self, index_qid: str, client: ElvClient, processor: SimpleProcessor, index: Index, ranker: SimpleRanker):
        self.client = client
        self.processor = processor
        self.index = index
        self.ranker = ranker
        self.index_qid = index_qid

    def search(self, args: SearchArgs) -> SearchOutput | ClipSearchOutput:
        if args["clips"]:
            # we need to use post instead of get for clip search, requires some changes to the clip-search args. 
            args = fix_clip_args(args)
        if 'num_retrieve' in args:
            num_retrieve = args['num_retrieve']
        else:
            num_retrieve = config.RETRIEVAL_NUM
        with timeit("Embedding query and processing it"):
            query = self.processor.process_query(self.index_qid, args)
        if args["uids"]:
            # then we are skipping the retrieval step and just ranking the provided uids
            uids = args["uids"]
        else:
            # retrieve the uids from the index
            with timeit("Retrieving document uids from vector index"):
                uids_per_field = [self.index.search(query["embedding"], field, k=num_retrieve) for field in args['search_fields']]
                # concatenate the results from different fields and deduplicate
                uids = set(reduce(lambda x, y: x+y, uids_per_field))
            with timeit("Retrieving document uids from text-based index"):
                # Also search the top results from the fabric search and perform ranking on these. 
                fs_args = {
                    "terms": query["corrected_query"],
                    "semantic": True,
                    "max_total": num_retrieve,
                    "limit": num_retrieve,
                    "display_fields": ["uid"],
                }
                r = self.client.search(object_id=self.index_qid, query=fs_args)
                uids.update(x["fields"]["uid"][0] for x in r["results"])
        with timeit("Ranking documents"):
            ranked_uids = self.ranker.rank(uids, args['max_total'], query)
        # delegate search formatting and additional features to fabric search engine
        with timeit("Searching uids on fabric"):
            del args['terms']
            res = self._search_uids([x[0] for x in ranked_uids], args)
        if not args["clips"]:
            res = SearchOutput().load(res)
            for rr, sr in zip(ranked_uids, res['results']):
                sr['score'] = rr[1]
        else:
            res = ClipSearchOutput().load(res)
        if "debug" in args:
            res['debug'] = {}
            if "weights" in query:
                res['debug'] = {"query weights": query["weights"]}
        return res

    # NOTE: if we want to build a new searcher we can move this method out of this class so others can call it. 
    def _search_uids(self, uids: List[str], args: SearchArgs) -> Any:
        # Select the documents from index based on uids, and rank them based on the provided order
        query = ' '.join(f'uid:\"{uid}\"^{idx+1}' for idx, uid in enumerate(reversed(uids)))
        if "filters" in args and args["filters"] != "":
            query = f"({query}) AND ({args['filters']})"
        logging.info(f"Querying fabric with {query}")
        args["filters"] = query
        return self.client.search(object_id=self.index_qid, query=args)

    def _uid_from_result(self, res: Dict[str, str]) -> str:
        return f"{res['hash']}{res['prefix']}"

# We make search query using post instead of get which means we need to pass the clip options differently due to the way the clip search post handler is implemented in qfab
def fix_clip_args(args: SearchArgs) -> SearchArgs:
    args = args.copy()
    clip_args = ["clips_include_source_tags", "clips_padding", "clips_max_duration", "clips_offering", "clips_coalescing_span"]
    # copy values from args to clip_options and remove 'clips_' prefix
    clip_options = {arg[6:]: args[arg] for arg in clip_args if arg in args}
    args["clip_options"] = clip_options
    return args
