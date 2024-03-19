from hyperopt import hp
import argparse

from experiments.experiments import get_default_experiment, get_trial_runner, run_experiment
from src.update import GlobalGitBuilder

def optimize_global_git(qid: str, data_path: str, auth: str, samples: int):
    experiment = get_default_experiment(qid, data_path, auth)

    experiment.index_builder = GlobalGitBuilder(experiment.index_builder.encoder)
    experiment.setup = lambda params: experiment.index_builder.encoder.set_k(int(params["K"]))

    space = {
        "K": hp.quniform('y', 100, 5000, 25)
    }

    run_trial = get_trial_runner(experiment)

    run_experiment(run_trial, space, samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    
    optimize_global_git(args.qid, args.data, args.auth, args.samples)