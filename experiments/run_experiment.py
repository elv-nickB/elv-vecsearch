import pandas as pd
import shutil
import os
from elv_client_py import ElvClient
import logging
import tempfile
from typing import Dict, Any, Callable

from experiments.utils import get_score, convert
from src.utils import timeit
from src.index.faiss import Index
from src.update.builder import IndexBuilder
from src.search.simple import SimpleSearcher
from src.ranking.rank import Ranker
from src.query_processing.simple import QueryProcessor
from src.format import SearchArgs
from src import config

class Experiment:
    def __init__(self, index: Index, index_builder: IndexBuilder, qid: str, client: ElvClient, processor: QueryProcessor, ranker: Ranker, data: pd.DataFrame, setup: Callable[[Dict[str, Any]], None]):
        self.index = index
        self.index_builder = index_builder
        self.qid = qid
        self.client = client
        self.processor = processor
        self.ranker = ranker
        self.data = data
        self.setup = setup

def get_experiment_runner(exp: Experiment):
    def run_experiment(params) -> float:
        exp.setup(params)
        tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
        with timeit("Building index"):
            exp.index_builder.build(exp.qid, exp.index, exp.client)
        searcher = SimpleSearcher(index_qid=exp.qid, client=exp.client, processor=exp.processor, index=exp.index, ranker=exp.ranker)
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
            res = searcher.search(args)
            # negate the score because we want to minimize the loss
            total += -get_score(res, data, q)

        shutil.rmtree(tmp_path)

        return total / len(queries)
    return run_experiment