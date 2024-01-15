
from format import *
import os
from typing import Dict, List, Tuple
from elv_client_py import ElvClient

from src import config
from src.index import FaissIndex 

def search(index_qid: str, args: SearchArgs, auth: str) -> SearchOutput:
    #SearchArgsFormat().validate(args)
    if not index_exists(index_qid):
        raise ValueError(f"Index for {index_qid} has not been built. Please build the index with /q/<qid>/update before searching.")
    index = FaissIndex.from_path(os.path.join(config.INDEX_PATH, index_qid))
    print(len(args['search_fields']), args['search_fields'])
    result = index.search(args['terms'], args['search_fields'], k2=args['max_total'])

    client = ElvClient.from_configuration_url(config.CONFIG_URL, auth)
    return fabric_search(index_qid, result, client, args)

def fabric_search(index_qid: str, uids: List[str], client: ElvClient, args: SearchArgs) -> SearchOutput:
    query = ' '.join(f'uid:\"{uid}\"' for uid in uids)
    args["terms"] = query
    return client.search(object_id=index_qid, query=args)

def index_exists(index_qid: str) -> bool:
    return os.path.exists(os.path.join(config.INDEX_PATH, index_qid))  
