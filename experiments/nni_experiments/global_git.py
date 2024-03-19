import argparse
import nni

from experiments.experiments import Experiment, get_trial_runner, get_default_experiment
from src import update

def optimize_global_git(qid: str, data: str, auth: str):
    experiment = get_default_experiment(qid, data, auth)
    experiment.index_builder = update.GlobalGitBuilder(experiment.index_builder.encoder)
    experiment.setup = lambda params: experiment.index_builder.set_k(int(params["K"]))
    run_trial = get_trial_runner(experiment)
    score = run_trial(nni.get_next_parameter())
    nni.report_final_result(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    
    optimize_global_git(args.qid, args.data, args.auth)