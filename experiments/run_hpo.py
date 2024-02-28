from src import update, config
from elv_client_py import ElvClient
import os
import tempfile
from typing import List, Callable
import numpy as np
from hyperopt import fmin, tpe, hp
import argparse
from sklearn.cluster import KMeans
import pandas as pd
import shutil

from src.index import FaissIndex
from src.embedding import VideoTagEncoder
from src.embedding import TextEncoder
from src.search import SimpleSearcher
from src.rank import SimpleRanker
from src.query_understanding import SimpleQueryProcessor
from src.format import SearchArgs, SearchOutput
from src.utils import LRUSearchCache

# TODO: implement
def get_score(res: SearchOutput, data: pd.DataFrame) -> float:
    return np.random.rand()

def _get_num_docs(content_id: str, client: ElvClient) -> int:
    res = client.content_object_metadata(object_id=content_id, metadata_subtree='indexer/stats/document/total')
    return int(res)

# converts uids of format iq_start_time_end_time into uid for fabric search which is based on hash + doc-prefix
def convert(uids: List[str], client: ElvClient, content_id: str) -> List[str]:
    ids = set(uid.split("|")[0] for uid in uids)
    # get num docs
    num_docs = _get_num_docs(content_id, client)
    args = {
        #"terms": "",
        "ids": ",".join(ids),
        "max_total": "-1",
        "limit": str(num_docs),
        "display_fields": "f_start_time,f_end_time"
    }
    res = client.search(object_id=content_id, query=args)
    start_end_times = [uid.split("|")[1].split('_') for uid in uids]
    start_end_times = [(int(start), int(end)) for start, end in start_end_times]
    uids = [doc["hash"]+doc["prefix"] for doc in res["results"] if (doc["fields"]["f_start_time"][0], doc["fields"]["f_end_time"][0]) in start_end_times]
    return uids

def optimize_git_clean():
    # initialize an elv-client where calls to the search api are cached for future use. This is helpful for building the index, 
    # where we need to make a single large request to the search api to get all the documents
    client = LRUSearchCache(ElvClient.from_configuration_url(config.CONFIG_URL, args.auth))
    def run_experiment(params) -> float:
        T = params["T"]
        K = int(params["K"])
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        index = FaissIndex(tmp_path, config.IndexConstructor)
        # takes in 2d numpy array and removes rows with similarity greater than T with another row
        # also uses k means to reduce the number of rows to K centroids
        encoder = VideoTagEncoder(config.SBERT_MODEL, K, T)
        processor = SimpleQueryProcessor(client, encoder)
        index_builder = update.IndexBuilder(encoder)
        index_builder.build(args.qid, index, client)
        ranker = SimpleRanker(index)
        searcher = SimpleSearcher(index_qid=args.qid, client=client, processor=processor, index=index, ranker=ranker)
        total = 0
        queries = data["query"].unique()
        for q in queries:
            uids = data[data["query"] == q]["iqShot"].to_list()
            uids = convert(uids, client, args.qid)
            args = {
                "search_fields": "f_object,f_speech_to_text,f_logo,f_celebrity,f_segment,f_display_title",
                "max_total": len(uids), 
                "limit": len(uids),
                "filters": ' '.join(f"uid:\"{uid}\"" for uid in uids), 
                "terms": q
            }
            args = SearchArgs().load(args)
            res = searcher.search(args)
            total += get_score(res, data)

        shutil.rmtree(tmp_path)

        return total / len(queries)
    space = {
        "T": hp.uniform('x', 0.6, 1.0),
        "K": hp.quniform('y', 1, 5, 1)
    }
    # we need to cache the encoder results so that building the index over and over doesn't take too long

    data = pd.read_csv(args.data)
    
    best = fmin(fn=run_experiment, space=space, algo=tpe.suggest, max_evals=args.samples)
    print(best)

def optimize_global_git():
    # initialize an elv-client where calls to the search api are cached for future use. This is helpful for building the index, 
    # where we need to make a single large request to the search api to get all the documents
    client = LRUSearchCache(ElvClient.from_configuration_url(config.CONFIG_URL, args.auth))
    encoder = VideoTagEncoder(config.SBERT_MODEL)
    qid = args.qid
    def run_experiment(params) -> float:
        K = int(params["K"])
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        index = FaissIndex(tmp_path, config.IndexConstructor)
        # takes in 2d numpy array and removes rows with similarity greater than T with another row
        # also uses k means to reduce the number of rows to K centroids
        processor = SimpleQueryProcessor(client, encoder)
        index_builder = update.GlobalGitBuilder(encoder, K)
        index_builder.build(qid, index, client)
        ranker = SimpleRanker(index)
        searcher = SimpleSearcher(index_qid=qid, client=client, processor=processor, index=index, ranker=ranker)
        total = 0
        queries = data["query"].unique()
        for q in queries:
            uids = data[data["query"] == q]["iqShot"].to_list()
            uids = convert(uids, client, qid)
            args = {
                "search_fields": "f_object,f_speech_to_text,f_logo,f_celebrity,f_segment,f_display_title",
                "max_total": len(uids), 
                "limit": len(uids),
                "filters": ' '.join(f"uid:\"{uid}\"" for uid in uids), 
                "terms": q
            }
            args = SearchArgs().load(args)
            res = searcher.search(args)
            total += get_score(res, data)

        shutil.rmtree(tmp_path)

        return total / len(queries)
    space = {
        "K": hp.quniform('y', 100, 500, 20)
    }
    data = pd.read_csv(args.data)

    best = fmin(fn=run_experiment, space=space, algo=tpe.suggest, max_evals=args.samples)
    print(best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    optimize_global_git()
