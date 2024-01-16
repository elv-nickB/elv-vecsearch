
from format import *
import os
from typing import List, Dict
from elv_client_py import ElvClient
import logging

from src import config
from src.index import FaissIndex 
from src.utils import timeit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search(index_qid: str, args: SearchArgs, auth: str) -> SearchOutput:
    if not index_exists(index_qid):
        raise ValueError(f"Index for {index_qid} has not been built. Please build the index with /q/<qid>/update before searching.")
    index = FaissIndex.from_path(os.path.join(config.INDEX_PATH, index_qid))
    # result is a list of document uids
    with timeit("retrieving document uids from vector index"):
        results = index.search(args['terms'], args['search_fields'], k2=args['max_total'])
    # for retrieving the original order of the documents
    pos_map = _get_pos_map(results)
    client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
    # delegate search formatting and additional features to fabric search engines
    with timeit("searching uids on fabric"):
        res = fabric_search(index_qid, [sr[1] for sr in results], client, args)
    res = SearchOutput().load(res)
    # Re-sort the results according to vector-based rankings
    with timeit("re-sorting the results"):
        res['results'].sort(key=lambda x: pos_map[_uid_from_result(x)])
    for vr, sr in zip(results, res['results']):
        sr['score'] = vr[0]
    return res

def fabric_search(index_qid: str, uids: List[str], client: ElvClient, args: SearchArgs) -> SearchOutput:
    # Select the documents from index based on uids
    query = ' '.join(f'uid:\"{uid}\"' for uid in uids)
    args["filters"] = query
    return client.search(object_id=index_qid, query=args)

# Does a vector index exist locally for the given index object id?
def index_exists(index_qid: str) -> bool:
    return os.path.exists(os.path.join(config.INDEX_PATH, index_qid))  

def _get_pos_map(l: List[str]) -> Dict[str, int]:
    return {l[i][1]: i for i in range(len(l))}

def _uid_from_result(res: Dict[str, str]) -> str:
    return f"{res['hash']}{res['prefix']}"
