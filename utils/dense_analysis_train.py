import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from analysis import *
from scipy.signal import savgol_filter

sns.set_style('whitegrid')

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_loss_curves(losses_dict, title, xlabel, ylabel, section, foldername, smoothing, filename=None, eval_cycle=None,
                     max_iter=None):
    plt.figure()

    for config, values in losses_dict.items():
        if smoothing:
            values = [savgol_filter(value, 121, 3) for value in values]

        mean, std = np.mean(values, axis=0), np.std(values, axis=0) / np.sqrt(len(values))

        if max_iter:
            mean, std = mean[:max_iter], std[:max_iter]
            plt.plot(np.arange(0, max_iter, 1), mean, label=config)
            plt.fill_between(np.arange(0, max_iter, 1), mean - std, mean + std, alpha=0.3)
        else:
            plt.plot(np.arange(len(mean)), mean, label=config)
            plt.fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)

    plt.xlim(left=30, right=len(mean))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join('imgs', section, foldername, '{}_{}_{}_{}.png'.format(filename, xlabel, ylabel, foldername)),
        dpi=700)
    plt.close()
    pass


def get_key_from_scalar_configs(configs, key):
    metrics_to_stack = [list(config['scalars'][key]) for config in configs]
    shortest_metric = min([len(m) for m in metrics_to_stack])
    return np.stack([metric[:shortest_metric] for metric in metrics_to_stack], axis=0)


def main():
    log_dir = 'C:\\Users\\Julien\\dense_case'

    experiment_configs = get_experiment_configs(log_dir=log_dir)

    metric_dict = {
        'wasserstein_distance__current_trajectory_vs_past_trajectories_': (
            'Current trajectory vs. Past trajectories', 'Training episode', None, 2500, True),
        'wasserstein_distance__current_trajectory_vs_uniform_disttribution_': (
            'Current trajectory vs. Uniform distribution', 'Training episode', None, 2500, True),
        'geometric_coverage': (
            'Geometric Coverage (gamma=0.9999)', 'Training episode', None, 2500, False),
        'simple_coverage': (
            'Simple Coverage', 'Training episode', None, 2500, False),
        'num_visited_sectors': (
            'Number visited sectors per episode', 'Training episode', None, 2500, True),
        'train_td_loss': (
            'TD-loss during training', 'Training episode', None, 2500, True),
        'intrinsic_episode_reward': (
            'Intrinsic reward per episode', 'Training episode', None, 2500, True),
        'occupancy_density_entropy': (
            'Occupancy Density Entropy', 'Training episode', None, 2500, False),
        'wasserstein_distance__cumulative_trajectory_vs_uniform_disttribution_': (
            'Cumulative trajectory vs. Uniform', 'Training episode', None, 2500, False),
        'train_episode_reward': (
            'Episode Reward', 'Training episode', None, 2500, True),
        'train_episode_length': (
            'Episode Length', 'Training episode', None, 2500, True),
        'val_episode_reward_mean': (
            'Validation Reward', 'Training episode', None, None, False),
    }

    # Check that geometric_coverage gamma was the same for all runs
    for config in experiment_configs:
        print(config['geometric_coverage_gamma'])

    extrinsic_intrinsic = find_matching_runs(experiment_configs, conditions={'intrinsic': True,
                                                                             'extrinsic': True})
    extrinsic_only = find_matching_runs(experiment_configs, conditions={'intrinsic': False,
                                                                        'extrinsic': True})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        extrinsic_intrinsic_metric = get_key_from_scalar_configs(extrinsic_intrinsic, metric_key)
        extrinsic_only_metric = get_key_from_scalar_configs(extrinsic_only, metric_key)
        plot_loss_curves(losses_dict={'Intrinsic + Extrinsic': extrinsic_intrinsic_metric,
                                      'Extrinsic': extrinsic_only_metric,
                                      },
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='vizdoom_vis_dense',
                         filename='ext_int_vs_only_ext', eval_cycle=eval_cycle, max_iter=max_iter,
                         smoothing=smoothing)


if __name__ == "__main__":
    main()
