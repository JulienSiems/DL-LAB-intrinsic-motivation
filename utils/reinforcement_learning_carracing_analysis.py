import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis import *

sns.set_style('whitegrid')


def plot_loss_curves(losses_dict, title, xlabel, ylabel, section, foldername, filename=None, eval_cycle=None,
                     max_iter=None):
    plt.figure()

    for config, values in losses_dict.items():
        mean, std = np.mean(values, axis=0), np.std(values, axis=0)
        if max_iter:
            plt.plot(np.arange(0, max_iter, eval_cycle), mean, label=config)
            plt.fill_between(np.arange(0, max_iter, eval_cycle), mean - std, mean + std, alpha=0.3)
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
    return np.stack([list(config['scalars'][key]) for config in configs], axis=0)


def main():
    log_dir = 'reinforcement_learning/carracing_report_2'

    experiment_configs = get_experiment_configs(log_dir=log_dir)

    metric_dict = {
        'val_episode_reward_mean': ('Validation Reward', 'Iteration', 50, 1000),
        'train_episode_reward': ('Training Reward', 'Iteration', None, None)
    }

    ddqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DDQN',
                                                                      'loss_function': 'L2',
                                                                      'soft_update': True})
    dqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DQN',
                                                                     'loss_function': 'L2',
                                                                     'soft_update': True})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter) in metric_dict.items():
        # Loss curve plots for resampling
        loss_resampling_false = get_key_from_scalar_configs(ddqn_configs, metric_key)
        loss_resampling_true = get_key_from_scalar_configs(dqn_configs, metric_key)
        plot_loss_curves(losses_dict={'DDQN': loss_resampling_false,
                                      'DQN': loss_resampling_true},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='carracing',
                         filename='l2_soft_update_true', eval_cycle=eval_cycle, max_iter=max_iter)

    ddqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DDQN',
                                                                      'loss_function': 'L1',
                                                                      'soft_update': True})
    dqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DQN',
                                                                     'loss_function': 'L1',
                                                                     'soft_update': True})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter) in metric_dict.items():
        # Loss curve plots for resampling
        loss_resampling_false = get_key_from_scalar_configs(ddqn_configs, metric_key)
        loss_resampling_true = get_key_from_scalar_configs(dqn_configs, metric_key)
        plot_loss_curves(losses_dict={'DDQN': loss_resampling_false,
                                      'DQN': loss_resampling_true},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='carracing',
                         filename='l1_soft_update_true', eval_cycle=eval_cycle, max_iter=max_iter)

    ddqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DDQN',
                                                                      'soft_update': False})
    dqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DQN',
                                                                     'soft_update': False})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter) in metric_dict.items():
        # Loss curve plots for resampling
        loss_resampling_false = get_key_from_scalar_configs(ddqn_configs, metric_key)
        loss_resampling_true = get_key_from_scalar_configs(dqn_configs, metric_key)
        plot_loss_curves(losses_dict={'DDQN': loss_resampling_false,
                                      'DQN': loss_resampling_true},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='carracing',
                         filename='soft_update_false', eval_cycle=eval_cycle, max_iter=max_iter)

    no_soft_update_configs = find_matching_runs(experiment_configs, conditions={'soft_update': False})
    soft_update_configs = find_matching_runs(experiment_configs, conditions={'soft_update': True})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter) in metric_dict.items():
        # Loss curve plots for resampling
        loss_resampling_false = get_key_from_scalar_configs(no_soft_update_configs, metric_key)
        loss_resampling_true = get_key_from_scalar_configs(soft_update_configs, metric_key)
        plot_loss_curves(losses_dict={'no soft update': loss_resampling_false,
                                      'soft update': loss_resampling_true},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='carracing',
                         filename='soft_update_true_false', eval_cycle=eval_cycle, max_iter=max_iter)

    ddqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DDQN',
                                                                      'loss_function': 'L2'})
    dqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DQN',
                                                                     'loss_function': 'L2'})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter) in metric_dict.items():
        # Loss curve plots for resampling
        loss_resampling_false = get_key_from_scalar_configs(ddqn_configs, metric_key)
        loss_resampling_true = get_key_from_scalar_configs(dqn_configs, metric_key)
        plot_loss_curves(losses_dict={'DDQN': loss_resampling_false,
                                      'DQN': loss_resampling_true},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='carracing',
                         filename='l2', eval_cycle=eval_cycle, max_iter=max_iter)

    ddqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DDQN',
                                                                      'loss_function': 'L1'})
    dqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DQN',
                                                                     'loss_function': 'L1'})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter) in metric_dict.items():
        # Loss curve plots for resampling
        loss_resampling_false = get_key_from_scalar_configs(ddqn_configs, metric_key)
        loss_resampling_true = get_key_from_scalar_configs(dqn_configs, metric_key)
        plot_loss_curves(losses_dict={'DDQN': loss_resampling_false,
                                      'DQN': loss_resampling_true},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='carracing',
                         filename='l1', eval_cycle=eval_cycle, max_iter=max_iter)

    ddqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DDQN'})
    dqn_configs = find_matching_runs(experiment_configs, conditions={'algorithm': 'DQN'})

    for metric_key, (metric_name, xlabel, eval_cycle, max_iter) in metric_dict.items():
        # Loss curve plots for resampling
        loss_resampling_false = get_key_from_scalar_configs(ddqn_configs, metric_key)
        loss_resampling_true = get_key_from_scalar_configs(dqn_configs, metric_key)
        plot_loss_curves(losses_dict={'DDQN': loss_resampling_false,
                                      'DQN': loss_resampling_true},
                         ylabel=metric_name,
                         xlabel=xlabel,
                         title=None, section='reinforcement_learning', foldername='carracing',
                         filename='ddqn_dqn', eval_cycle=eval_cycle, max_iter=max_iter)


if __name__ == "__main__":
    main()
