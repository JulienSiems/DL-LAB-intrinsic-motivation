import glob
import os
import json

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tf_event_file(tfevent_file):
    # Read in tensorflow summary file
    event_acc = EventAccumulator(path=tfevent_file)
    event_acc.Reload()
    run_config = {}
    for scalar_summary_key in event_acc.Tags()['scalars']:
        _, step_nums, values = zip(*event_acc.Scalars(scalar_summary_key))
        run_config[scalar_summary_key] = values

    return run_config


def get_experiment_configs(log_dir):
    summary_directories = glob.glob(os.path.join(log_dir, '*'))

    experiment_configs = []
    for directory in summary_directories:
        # Read tfevent summary files
        tfevent_file = glob.glob(os.path.join(directory, "*.tfevents.*"))
        run_config = read_tf_event_file(tfevent_file[0])

        # Read config file
        config_file = glob.glob(os.path.join(directory, "config.json"))[0]
        config = json.load(open(config_file, 'r'))
        config['scalars'] = run_config
        config['directory'] = directory
        experiment_configs.append(config)
    return experiment_configs


def find_matching_runs(configs, conditions):
    searched_config = []
    for config in configs:
        # Only select config if all conditions are satisfied
        conds_satisfied = [config[cond_key] == cond_val for cond_key, cond_val in conditions.items()]
        if all(conds_satisfied):
            searched_config.append(config)

    return searched_config


def main():
    log_dir = os.path.join('reinforcement_learning', 'carracing_2')
    experiment_configs = get_experiment_configs(log_dir)

    searched_config = find_matching_runs(configs=experiment_configs,
                                         conditions={'num_episodes': 1000})

if __name__ == "__main__":
    main()
