import argparse
from hyperopt import hp

from experiments.experiments import get_default_experiment, get_trial_runner, run_experiment

def optimize_git_clean(qid: str, data_path: str, auth: str, samples: int):
    experiment = get_default_experiment(qid, data_path, auth)
    
    space = {
        "T": hp.uniform('x', 0.95, 1.0),
        "K": hp.quniform('y', 4, 5, 1)
    }

    experiment.setup = lambda params: experiment.index_builder.encoder.set_t_k(params["T"], int(params["K"]))

    run_trial = get_trial_runner(experiment)
    
    run_experiment(run_trial, space, samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    
    optimize_git_clean(args.qid, args.data, args.auth, args.samples)