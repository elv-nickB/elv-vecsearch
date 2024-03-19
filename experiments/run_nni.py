from nni.experiment import Experiment
import argparse

def main():
    experiment = Experiment('local')

    search_space = {
        'K': {'_type': 'quniform', '_value': [1, 8, 1]},
        'T': {'_type': 'uniform', '_value': [0.7, 1]},
    }

    experiment.config.trial_command = f'python nni_experiments/{args.script} --auth {args.auth} --qid {args.qid} --samples {args.samples} --data {args.data}'
    experiment.config.trial_code_directory = 'experiments'

    experiment.config.search_space = search_space
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

    experiment.config.max_trial_number = args.samples
    experiment.config.trial_concurrency = 2

    experiment.run(8087)
    input('Press enter to quit')
    experiment.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=str, help="Authorization token", required=True)
    parser.add_argument("--qid", type=str, help="Index id", required=True)
    parser.add_argument("--samples", type=int, help="Number of samples to try", required=True)
    parser.add_argument("--script", type=str, help="Path of script to run in nni_experiments folder", required=True)
    parser.add_argument("--data", type=str, help="Path to csv file containing queries and their scores for evaluating the model", required=True)
    args = parser.parse_args()
    
    main()