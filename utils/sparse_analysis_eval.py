import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from analysis import *
from scipy.signal import savgol_filter

sns.set_style('whitegrid')


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

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join('imgs', section, foldername, '{}_{}_{}_{}.pdf'.format(filename, xlabel, ylabel, foldername)))
    plt.close()
    pass


def get_key_from_scalar_configs(configs, key):
    metrics_to_stack = [list(config['scalars'][key]) for config in configs]
    shortest_metric = min([len(m) for m in metrics_to_stack])
    return np.stack([metric[:shortest_metric] for metric in metrics_to_stack], axis=0)


def main():
    log_dir = 'C:\\Users\\Julien\\sparse_intrinsic_eval_metrics'

    experiment_configs = get_experiment_configs(log_dir=log_dir)

    metric_dict = {
        'exploration_cross_entropy': (
            'Exploration Cross Entropy', 'Training episode', None, None, False),
        'exploration_entropy_gain': (
            'Exploration Entropy Gain', 'Training episode', None, None, False),
        'exploration_entropy_score': (
            'Exploration Entropy Score', 'Training episode', None, None, False),
        'exploration_variance_total_variance': (
            'Exploration Variance (Total Variance)', 'Training episode', None, None, False),
        'exploration_variance_wasserstein_variance': (
            'Exploration Variance (Wasserstein Distance)', 'Training episode', None, None, False),
    }

    # Check that geometric_coverage gamma was the same for all runs
    for config in experiment_configs:
        print(config['geometric_coverage_gamma'])

    intrinsic_duelling_true = find_matching_runs(experiment_configs, conditions={'intrinsic': True,
                                                                                 'extrinsic': False,
                                                                                 'duelling': True})
    intrinsic_duelling_false = find_matching_runs(experiment_configs, conditions={'intrinsic': True,
                                                                                  'extrinsic': False,
                                                                                  'duelling': False})
    extrinsic_duelling_true = find_matching_runs(experiment_configs, conditions={'intrinsic': False,
                                                                                 'extrinsic': True,
                                                                                 'duelling': True})
    extrinsic_duelling_false = find_matching_runs(experiment_configs, conditions={'intrinsic': False,
                                                                                  'extrinsic': True,
                                                                                  'duelling': False})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
        # Loss curve plots for resampling
        intrinsic_duelling_true_metric = get_key_from_scalar_configs(intrinsic_duelling_true, metric_key)
        intrinsic_duelling_false_metric = get_key_from_scalar_configs(intrinsic_duelling_false, metric_key)
        extrinsic_duelling_true_metric = get_key_from_scalar_configs(extrinsic_duelling_true, metric_key)
        # extrinsic_duelling_false_metric = get_key_from_scalar_configs(extrinsic_duelling_false, metric_key)

        plot_loss_curves(losses_dict={'Int. (duel.)': intrinsic_duelling_true_metric,
                                      'Int. (no duel.)': intrinsic_duelling_false_metric,
                                      'Ext. (duel.)': extrinsic_duelling_true_metric,
                                      # 'Ext. (no duel.)': extrinsic_duelling_false_metric
                                      },
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='vizdoom_vis_eval',
                         filename='ext_int_duelling', eval_cycle=eval_cycle, max_iter=max_iter, smoothing=smoothing)

        for metric_key, (metric_name, xlabel, eval_cycle, max_iter, smoothing) in metric_dict.items():
            # Loss curve plots for resampling
            intrinsic_duelling_true_metric = get_key_from_scalar_configs(intrinsic_duelling_true, metric_key)
            intrinsic_duelling_false_metric = get_key_from_scalar_configs(intrinsic_duelling_false, metric_key)
            extrinsic_duelling_true_metric = get_key_from_scalar_configs(extrinsic_duelling_true, metric_key)
            # extrinsic_duelling_false_metric = get_key_from_scalar_configs(extrinsic_duelling_false, metric_key)

            plot_loss_curves(losses_dict={'Int. (duel.)': intrinsic_duelling_true_metric,
                                          'Int. (no duel.)': intrinsic_duelling_false_metric,
                                          # 'Ext. (duel.)': extrinsic_duelling_true_metric,
                                          # 'Ext. (no duel.)': extrinsic_duelling_false_metric
                                          },
                             ylabel=metric_name,
                             xlabel=xlabel,
                             title=None, section='reinforcement_learning', foldername='vizdoom_vis_eval',
                             filename='int_vs_int_duelling', eval_cycle=eval_cycle, max_iter=max_iter, smoothing=smoothing)


if __name__ == "__main__":
    main()
