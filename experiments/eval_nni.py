from src import update, config
from elv_client_py import ElvClient
import tempfile
import argparse
import pandas as pd
import shutil
import os
import logging
import nni

from src.index import FaissIndex
from src.embedding import VideoTagEncoder
from src.search import SimpleSearcher
from src.rank import SimpleRanker
from src.query_understanding import SimpleQueryProcessor
from src.format import SearchArgs
from src.utils import LRUSearchCache
from experiments.utils import get_score, convert
from experiments.run_experiment import Experiment, get_experiment_runner
from src.utils import timeit

def main():
    client = LRUSearchCache(ElvClient.from_configuration_url(config.CONFIG_URL, args.auth))
    encoder = VideoTagEncoder(config.SBERT_MODEL)
    cache_path = os.path.join(config.DATA_PATH, 'experiments', 'embedding_cache', args.qid)
    if os.path.exists(cache_path):
        logging.info(f"Loading cache from {cache_path}")
        encoder.load_cache(cache_path)
    tmp_path = tempfile.mkdtemp(dir=config.TMP_PATH)
    processor = SimpleQueryProcessor(client, encoder)
    index = FaissIndex(tmp_path, config.IndexConstructor)
    ranker = SimpleRanker(index)
    index_builder = update.IndexBuilder(encoder)
    qid = args.qid
    data = pd.read_csv(os.path.join(config.SRC_PATH, '..', args.data))
    experiment = Experiment(index, index_builder, qid, client, processor, ranker, data, lambda params: encoder.set_t_k(params["T"], params["K"]))
    run_experiment = get_experiment_runner(experiment)
    params = nni.get_next_parameter()
    score = run_experiment(params)
    nni.report_final_result(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    
    main()