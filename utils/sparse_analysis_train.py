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
            values = [savgol_filter(value, 81, 3) for value in values]

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
    log_dir = 'C:\\Users\\Julien\\sparse_intrinsic_evaluation'

    experiment_configs = get_experiment_configs(log_dir=log_dir)

    metric_dict = {
        'wasserstein_distance__current_trajectory_vs_past_trajectories_': (
            'Current trajectory vs. Past trajectories', 'Training episode', None, 1250, True),
        'wasserstein_distance__current_trajectory_vs_uniform_disttribution_': (
            'Current trajectory vs. Uniform distribution', 'Training episode', None, 1250, True),
        'geometric_coverage': (
            'Geometric Coverage (gamma=0.9999)', 'Training episode', None, 1750, False),
        'simple_coverage': (
            'Simple Coverage', 'Training episode', None, 1250, False),
        'num_visited_sectors': (
            'Number visited sectors per episode', 'Training episode', None, 1250, True),
        # 'train_td_loss': (
        #    'TD-loss during training', 'Training episode', None, 1750, True),
        # 'intrinsic_episode_reward': (
        #    'Intrinsic reward per episode', 'Training episode', None, 1750, True),
        'occupancy_density_entropy': (
            'Occupancy Density Entropy', 'Training episode', None, 1250, False),
        'wasserstein_distance__cumulative_trajectory_vs_uniform_disttribution_': (
            'Cumulative trajectory vs. Uniform', 'Training episode', None, 1250, False),
    }

    # Check that geometric_coverage gamma was the same for all runs
    for config in experiment_configs:
        print(config['geometric_coverage_gamma'])

    intrinsic_duelling_true = find_matching_runs(experiment_configs, conditions={'intrinsic': True,
                                                                                 'extrinsic': False,
                                                                                 'duelling': True,
                                                                                 'epsilon': 0.1,
                                                                                 'mu_intrinsic': .0005,
                                                                                 'iqn': False,
                                                                                 'experience_replay': 'Uniform'})

    intrinsic_duelling_false = find_matching_runs(experiment_configs, conditions={'intrinsic': True,
                                                                                  'extrinsic': False,
                                                                                  'duelling': False,
                                                                                  'epsilon': 0.1,
                                                                                  'mu_intrinsic': .0005,
                                                                                  'iqn': False,
                                                                                  'experience_replay': 'Uniform'})

    intrinsic_duelling_false_higher_mu = find_matching_runs(experiment_configs, conditions={'intrinsic': True,
                                                                                            'extrinsic': False,
                                                                                            'duelling': False,
                                                                                            'epsilon': 0.1,
                                                                                            'mu_intrinsic': 1.0,
                                                                                            'iqn': False,
                                                                                            'experience_replay': 'Uniform'})
    intrinsic_duelling_true_higher_mu = find_matching_runs(experiment_configs, conditions={'intrinsic': True,
                                                                                           'extrinsic': False,
                                                                                           'duelling': True,
                                                                                           'epsilon': 0.1,
                                                                                           'mu_intrinsic': 1.0,
                                                                                           'iqn': False,
                                                                                           'experience_replay': 'Uniform'})

    extrinsic_duelling_true = find_matching_runs(experiment_configs, conditions={'intrinsic': False,
                                                                                 'extrinsic': True,
                                                                                 'duelling': True,
                                                                                 'epsilon': 0.1,
                                                                                 'mu_intrinsic': .0005,
                                                                                 'iqn': False,
                                                                                 'experience_replay': 'Uniform'})

    extrinsic_duelling_false = find_matching_runs(experiment_configs, conditions={'intrinsic': False,
                                                                                  'extrinsic': True,
                                                                                  'duelling': False,
                                                                                  'epsilon': 0.1,
                                                                                  'mu_intrinsic': .0005,
                                                                                  'iqn': False,
                                                                                  'experience_replay': 'Uniform'})
    random_search = find_matching_runs(experiment_configs, conditions={'epsilon': 1.0})

    iqn = find_matching_runs(experiment_configs, conditions={'iqn': True})
    prioritized_exp = find_matching_runs(experiment_configs, conditions={'experience_replay': 'Prioritized'})

    '''
    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        prioritized_exp_metric = get_key_from_scalar_configs(prioritized_exp, metric_key)
        intrinsic_duelling_false_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_false_higher_mu,
                                                                                metric_key)

        plot_loss_curves(losses_dict={'Prioritized Experience Replay': prioritized_exp_metric,
                                      'Uniform Experience Replay': intrinsic_duelling_false_higher_mu_metric},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='vizdoom_vis',
                         filename='prio', eval_cycle=eval_cycle, max_iter=max_iter,
                         smoothing=smoothing)

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        iqn_metric = get_key_from_scalar_configs(iqn, metric_key)
        intrinsic_duelling_false_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_false_higher_mu,
                                                                                metric_key)

        plot_loss_curves(losses_dict={'IQN': iqn_metric,
                                      'No IQN': intrinsic_duelling_false_higher_mu_metric},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='vizdoom_vis',
                         filename='iqn', eval_cycle=eval_cycle, max_iter=max_iter,
                         smoothing=smoothing)

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        prioritized_exp_metric = get_key_from_scalar_configs(prioritized_exp, metric_key)
        intrinsic_duelling_false_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_false_higher_mu,
                                                                                metric_key)

        plot_loss_curves(losses_dict={'Prioritized Experience Replay': prioritized_exp_metric,
                                      'Uniform Experience Replay': intrinsic_duelling_false_higher_mu_metric},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='vizdoom_vis',
                         filename='prio', eval_cycle=eval_cycle, max_iter=max_iter,
                         smoothing=smoothing)

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        intrinsic_duelling_true_metric = get_key_from_scalar_configs(intrinsic_duelling_true, metric_key)
        intrinsic_duelling_false_metric = get_key_from_scalar_configs(intrinsic_duelling_false, metric_key)
        extrinsic_duelling_true_metric = get_key_from_scalar_configs(extrinsic_duelling_true, metric_key)
        extrinsic_duelling_false_metric = get_key_from_scalar_configs(extrinsic_duelling_false, metric_key)
        intrinsic_duelling_false_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_false_higher_mu,
                                                                                metric_key)
        intrinsic_duelling_true_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_true_higher_mu,
                                                                                metric_key)
        plot_loss_curves(losses_dict={'eta=0.0005': intrinsic_duelling_false_metric,
                                      'eta=1.0': intrinsic_duelling_false_higher_mu_metric},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='vizdoom_vis',
                         filename='int_high_vs_int_low', eval_cycle=eval_cycle, max_iter=max_iter,
                         smoothing=smoothing)
    '''
    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        intrinsic_duelling_true_metric = get_key_from_scalar_configs(intrinsic_duelling_true, metric_key)
        intrinsic_duelling_false_metric = get_key_from_scalar_configs(intrinsic_duelling_false, metric_key)
        extrinsic_duelling_true_metric = get_key_from_scalar_configs(extrinsic_duelling_true, metric_key)
        extrinsic_duelling_false_metric = get_key_from_scalar_configs(extrinsic_duelling_false, metric_key)
        intrinsic_duelling_false_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_false_higher_mu,
                                                                                metric_key)
        intrinsic_duelling_true_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_true_higher_mu,
                                                                               metric_key)
        plot_loss_curves(losses_dict={'Dueling': intrinsic_duelling_true_higher_mu_metric,
                                      'No Dueling': intrinsic_duelling_false_higher_mu_metric,
                                      },
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='vizdoom_vis',
                         filename='int_duel_vs_non_duel', eval_cycle=eval_cycle, max_iter=max_iter,
                         smoothing=smoothing)

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        intrinsic_duelling_true_metric = get_key_from_scalar_configs(intrinsic_duelling_true, metric_key)
        intrinsic_duelling_false_metric = get_key_from_scalar_configs(intrinsic_duelling_false, metric_key)
        extrinsic_duelling_true_metric = get_key_from_scalar_configs(extrinsic_duelling_true, metric_key)
        extrinsic_duelling_false_metric = get_key_from_scalar_configs(extrinsic_duelling_false, metric_key)
        random_search_metric = get_key_from_scalar_configs(random_search, metric_key)
        intrinsic_duelling_false_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_false_higher_mu,
                                                                                metric_key)
        plot_loss_curves(losses_dict={  # 'Int. (duel.)': intrinsic_duelling_true_metric,
            'Intrinsic': intrinsic_duelling_false_higher_mu_metric,
            # 'Intrinsic': intrinsic_duelling_false_metric,
            # 'Ext. (duel.)': extrinsic_duelling_true_metric,
            'No Reward': extrinsic_duelling_false_metric,
            'Random Trajectory': random_search_metric
        },
            ylabel=metric_name,
            xlabel=xlabel,
            title=None, section='reinforcement_learning', foldername='vizdoom_vis',
            filename='ext_vs_int_with_random', eval_cycle=eval_cycle, max_iter=max_iter, smoothing=smoothing)

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        intrinsic_duelling_true_metric = get_key_from_scalar_configs(intrinsic_duelling_true, metric_key)
        intrinsic_duelling_false_metric = get_key_from_scalar_configs(intrinsic_duelling_false, metric_key)
        extrinsic_duelling_true_metric = get_key_from_scalar_configs(extrinsic_duelling_true, metric_key)
        extrinsic_duelling_false_metric = get_key_from_scalar_configs(extrinsic_duelling_false, metric_key)
        random_search_metric = get_key_from_scalar_configs(random_search, metric_key)
        intrinsic_duelling_false_higher_mu_metric = get_key_from_scalar_configs(intrinsic_duelling_false_higher_mu,
                                                                                metric_key)
        plot_loss_curves(losses_dict={  # 'Int. (duel.)': intrinsic_duelling_true_metric,
            'Intrinsic': intrinsic_duelling_false_higher_mu_metric,
            # 'Intrinsic': intrinsic_duelling_false_metric,
            # 'Ext. (duel.)': extrinsic_duelling_true_metric,
            'No Reward': extrinsic_duelling_false_metric,
            # 'Random Trajectory': random_search_metric},
            },
            ylabel=metric_name,
            xlabel=xlabel,
            title=None, section='reinforcement_learning', foldername='vizdoom_vis',
            filename='ext_vs_int', eval_cycle=eval_cycle, max_iter=max_iter, smoothing=smoothing)


if __name__ == "__main__":
    main()
