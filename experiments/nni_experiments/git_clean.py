import argparse
import nni

from experiments.experiments import Experiment, get_trial_runner, get_default_experiment

def optimize_git_clean(qid: str, data: str, auth: str):
    experiment = get_default_experiment(qid, data, auth)
    params = nni.get_next_parameter()
    experiment.setup = lambda params: experiment.index_builder.encoder.set_t_k(params["T"], int(params["K"]))
    run_trial = get_trial_runner(experiment)
    score = run_trial(params)
    nni.report_final_result(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    
    optimize_git_clean(args.qid, args.data, args.auth)