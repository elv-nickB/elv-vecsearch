from src import config
from elv_client_py import ElvClient
import tempfile
from hyperopt import fmin, tpe, hp, Trials
import argparse
import pandas as pd
import shutil
import os
import logging

from src.index.faiss import FaissIndex
from src.embedding import VideoTagEncoder
from src.search.simple import SimpleSearcher
from src.ranking.simple import SimpleRanker
from src.query_processing.simple import SimpleQueryProcessor
from src.format import SearchArgs
from src.update import builder
from src.utils import LRUSearchCache
from experiments.utils import get_score, convert
from src.utils import timeit
from experiments.run_experiment import Experiment, get_experiment_runner

def main():
    # initialize an elv-client where calls to the search api are cached for future use. This is helpful for building the index, 
    # where we need to make a single large request to the search api to get all the documents
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
    index_builder = builder.IndexBuilder(encoder)
    qid = args.qid
    data = pd.read_csv(args.data)
    experiment = Experiment(index, index_builder, qid, client, processor, ranker, data, lambda params: encoder.set_t_k(params["T"], int(params["K"])))
    
    space = {
        "T": hp.uniform('x', 0.6, 1.0),
        "K": hp.quniform('y', 1, 5, 1)
    }

    run_experiment = get_experiment_runner(experiment)
    
    trials = Trials()
    fmin(fn=run_experiment, space=space, algo=tpe.suggest, max_evals=args.samples, trials=trials)
    # Accessing the results
    for trial in trials.trials:
        print(trial['result'], trial['misc']['vals'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    
    main()