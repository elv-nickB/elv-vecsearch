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
from loguru import logger
from collections import defaultdict


from src.index import FaissIndex
from src.embedding import get_encoder_with_cache
from src.embedding import TextEncoder
from src.search import SimpleSearcher
from src.rank import SimpleRanker
from src.query_understanding import SimpleQueryProcessor
from src.format import SearchArgs, SearchOutput
from src.utils import LRUSearchCache
import src.loss as LossFunc


def get_loss(res: SearchOutput, data: pd.DataFrame, query, k=20, reRank=False, useRank=True) -> float:
    _df = data.loc[data['query'] == query]
    iqShot2rating = {k: v for k, v in zip(_df['iqShot'], _df['avg_rating'])}
    ranking, simiScore, rating = [], [], []
    newTopM = defaultdict(list)
    if 'results' not in res:
        logger.info('Cannot pass search results')
        return None
    for rs in res['results']:
        if len(ranking) >= k:
            break
        try:
            _iqShot = rs['hash'] + '|' + str(rs['fields']['f_start_time'][-1]) + '_' + str(
                rs['fields']['f_end_time'][-1])
            _rank = rs['rank']
            _score = rs['score']
        except KeyError as e:
            logger.info('search results missing key', e)
        else:
            if _iqShot in iqShot2rating:
                ranking.append(_rank)
                simiScore.append(_score)
                rating.append(iqShot2rating[_iqShot])
            newTopM[_iqShot].append({
                'query': query,
                'rank': _rank,
                'similarity score': _score,
            })
    # TODO: where to dump the newTopM
    y_true = np.array(rating)
    pred_score = np.array(simiScore)
    pred_rank = np.array(ranking)
    return LossFunc.pointwise_regression_error(y_true, pred_score, pred_rank, topk=k, reRank=reRank, useRank=useRank)


def _get_num_docs(content_id: str, client: ElvClient) -> int:
    res = client.content_object_metadata(
        object_id=content_id, metadata_subtree='indexer/stats/document/total')
    return int(res)

# converts uids of format iq_start_time_end_time into uid for fabric search which is based on hash + doc-prefix


def convert(uids: List[str], client: ElvClient, content_id: str) -> List[str]:
    ids = set(uid.split("|")[0] for uid in uids)
    # get num docs
    num_docs = _get_num_docs(content_id, client)
    args = {
        # "terms": "",
        "ids": ",".join(ids),
        "max_total": "-1",
        "limit": str(num_docs),
        "display_fields": "f_start_time,f_end_time"
    }
    res = client.search(object_id=content_id, query=args)
    start_end_times = [uid.split("|")[1].split('_') for uid in uids]
    start_end_times = [
        (int(start), int(end)) for start, end in start_end_times]
    uids = [doc["hash"]+doc["prefix"] for doc in res["results"]
            if (doc["fields"]["f_start_time"][0], doc["fields"]["f_end_time"][0]) in start_end_times]
    return uids

# Returns a function which evaluates a given set of parameters. We want to optimize this function.


def get_evaluation(qid: str, auth: str, encoder: TextEncoder, data: pd.DataFrame) -> Callable[[dict], float]:
    path = os.path.join(config.DATA_PATH, "experiments")
    # initialize an elv-client where calls to the search api are cached for future use. This is helpful for building the index,
    # where we need to make a single large request to the search api to get all the documents
    client = LRUSearchCache(
        ElvClient.from_configuration_url(config.CONFIG_URL, auth))
    processor = SimpleQueryProcessor(client, encoder)

    def run_experiment(params) -> float:
        T = params["T"]
        K = int(params["K"])
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        index = FaissIndex(tmp_path, config.IndexConstructor)
        # takes in 2d numpy array and removes rows with similarity greater than T with another row
        # also uses k means to reduce the number of rows to K centroids

        def postprocess(x: np.ndarray) -> np.ndarray:
            to_remove = []
            for i in range(x.shape[0]):
                for j in range(i+1, x.shape[0]):
                    if np.dot(x[i], x[j]) > T:
                        to_remove.append(j)
            x = np.delete(x, to_remove, axis=0)
            if x.shape[0] > K:
                kmeans = KMeans(n_clusters=K, n_init=10)
                kmeans.fit(x)
                x = kmeans.cluster_centers_
            return x

        index_buider = update.IndexBuilder(encoder, postprocess)
        index_buider.build(qid, index, client)
        index.set_path(os.path.join(path, qid, f"{T}_{K}"))
        ranker = SimpleRanker(index)
        searcher = SimpleSearcher(
            index_qid=qid, client=client, processor=processor, index=index, ranker=ranker)
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
            total += get_loss(res, data, q)

        return total / len(queries)
    return run_experiment


def main():
    space = {
        "T": hp.uniform('x', 0, 0.5),
        "K": hp.quniform('y', 1, 5, 1)
    }
    # we need to cache the encoder results so that building the index over and over doesn't take too long
    encoder = get_encoder_with_cache(config.SBERT_MODEL)
    data = pd.read_csv(args.data)
    evaluator = get_evaluation(args.qid, args.auth, encoder, data)

    best = fmin(fn=evaluator, space=space,
                algo=tpe.suggest, max_evals=args.samples)
    print(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str,
                        help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int,
                        help="Number of samples to try", required=True)
    parser.add_argument(
        "--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    main()
