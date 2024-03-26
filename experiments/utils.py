import numpy as np
from typing import List
import pandas as pd
from elv_client_py import ElvClient
from sklearn.metrics import ndcg_score
import logging
from collections import defaultdict

from src.format import SearchOutput
import experiments.loss as LossFunc


def get_loss(res: SearchOutput, data: pd.DataFrame, query, k=20, reRank=False, useRank=True) -> float:
    _df = data.loc[data['query'] == query]
    iqShot2rating = {k: v for k, v in zip(_df['iqShot'], _df['avg_rating'])}
    ranking, simiScore, rating = [], [], []
    newTopM = defaultdict(list)
    if 'results' not in res:
        logging.info('Cannot pass search results')
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
            logging.info('search results missing key', e)
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

def get_score(res: SearchOutput, data: pd.DataFrame, query: str) -> float:
    examples = data[data["query"] == query]
    true_scores, predicted_scores = [], []
    for r in res['results']:
        doc_id = f'{r["id"]}|{r["fields"]["f_start_time"][0]}_{r["fields"]["f_end_time"][0]}'
        if doc_id not in examples["iqShot"].values:
            logging.error(f"Could not find {doc_id} in examples, query: {query}")
            continue
        else:
            true_scores.append(examples[examples["iqShot"] == doc_id]["avg_rating"].values[0])
            predicted_scores.append(r["score"])
    return ndcg_score([true_scores], [predicted_scores])

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
    id_time = [uid.split("|") for uid in uids]
    doc_ids = [(id, int(time.split('_')[0]), int(time.split('_')[1])) for id, time in id_time]
    uids = [doc["hash"]+doc["prefix"] for doc in res["results"] if (doc["id"], doc["fields"]["f_start_time"][0], doc["fields"]["f_end_time"][0]) in doc_ids]
    return uids
