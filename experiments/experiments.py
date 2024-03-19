
from src import update, config
from elv_client_py import ElvClient
import tempfile
from typing import Dict, Any, Callable
from hyperopt import fmin, tpe, Trials
import pandas as pd
import shutil
import os
import logging

from src.index import FaissIndex, Index
from src.embedding import VideoTagEncoder
from src.search import Searcher, SimpleSearcher
from src.rank import SimpleRanker
from src.update import IndexBuilder
from src.query_understanding import SimpleQueryProcessor
from src.format import SearchArgs
from src.utils import LRUSearchCache
from experiments.utils import get_score, convert, get_loss
from src.utils import timeit

class Experiment:
    def __init__(self, 
            searcher: Searcher,
            index_builder: IndexBuilder, 
            index: Index,
            qid: str, 
            client: ElvClient,
            data: pd.DataFrame, 
            # setup function that takes a dictionary of parameters and sets up the experiment state
            setup: Callable[[Dict[str, Any]], None]):
        self.searcher = searcher
        self.qid = qid
        self.client = client
        self.index_builder = index_builder
        self.index = index
        self.data = data
        self.setup = setup

def get_trial_runner(exp: Experiment):
    def run_trial(params) -> float:
        print(params)
        exp.setup(params)
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        with timeit("Building index"):
            exp.index_builder.build(exp.qid, exp.index, exp.client)
        data = exp.data
        total = 0
        queries = data["query"].unique()
        for q in queries:
            uids = data[data["query"] == q]["iqShot"].to_list()
            uids = convert(uids, exp.client, exp.qid)
            if len(uids) <= 1:
                logging.warning(f"Skipping query {q} because it has less than 2 examples")
                continue
            args = {
                "search_fields": "f_object,f_speech_to_text,f_logo,f_celebrity,f_segment,f_display_title",
                "max_total": len(uids), 
                "limit": len(uids),
                "uids": ','.join(uids), 
                "terms": q
            }
            args = SearchArgs().load(args)
            res = exp.searcher.search(args)
            total += get_loss(res, data, q)

        shutil.rmtree(tmp_path)

        return total / len(queries)
    return run_trial

def get_default_experiment(qid: str, data_path: str, auth: str) -> Experiment:
    client = LRUSearchCache(ElvClient.from_configuration_url(config.CONFIG_URL, auth))
    encoder = VideoTagEncoder(config.SBERT_MODEL)
    cache_path = os.path.join(config.DATA_PATH, 'experiments', 'embedding_cache', qid)
    if os.path.exists(cache_path):
        logging.info(f"Loading cache from {cache_path}")
        encoder.load_cache(cache_path)
    tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
    processor = SimpleQueryProcessor(client, encoder)
    index = FaissIndex(tmp_path, config.IndexConstructor)
    ranker = SimpleRanker(index)
    index_builder = update.IndexBuilder(encoder)
    searcher = SimpleSearcher(qid, client, processor, index, ranker)
    data = pd.read_csv(os.path.join(config.PROJECT_PATH, data_path))
    experiment = Experiment(searcher, index_builder, index, qid, client, data, lambda _: None)
    return experiment

def run_experiment(trial_fn: Callable[[Dict[str, Any]], float], space: Dict[str, Any], samples: int):
    trials = Trials()
    fmin(fn=trial_fn, space=space, algo=tpe.suggest, max_evals=samples, trials=trials)
    # Accessing the results
    for trial in trials.trials:
        print(trial['result'], trial['misc']['vals'])