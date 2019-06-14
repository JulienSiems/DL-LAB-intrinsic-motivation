from __future__ import print_function

import json
import os
import glob

import click
import gym
import numpy as np
import torch
from datetime import datetime
from agent.dqn_agent import DQNAgent
from agent.networks import *
from agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork
from train_carracing import run_episode

np.random.seed(0)

device = torch.device('cpu')


@click.command()
@click.option('-mp', '--model_path',
              default='carracing_report_2',
              type=click.STRING)
@click.option('-nte', '--n_test_episodes', default=15, type=click.INT)
def main(model_path, n_test_episodes):
    run_paths = glob.glob(os.path.join(model_path, '*'))
    for run_path in run_paths:
        if len(glob.glob(os.path.join(run_path, 'carracing_results*'))) > 0:
            print(run_path, 'already processed')
            continue
        # Load run config
        run_config = json.load(open(os.path.join(run_path, 'config.json'), 'r'))
        env = gym.make("CarRacing-v0").unwrapped

        num_actions = 5

        # Define networks and load agent
        if run_config['model'] == 'Resnet':
            Q_net = ResnetVariant(num_actions=num_actions, history_length=run_config['history_length'] + 1).to(device)
            Q_target_net = ResnetVariant(num_actions=num_actions, history_length=run_config['history_length'] + 1).to(
                device)
        elif run_config['model'] == 'Lenet':
            Q_net = LeNetVariant(num_actions=num_actions, history_length=run_config['history_length'] + 1).to(device)
            Q_target_net = LeNetVariant(num_actions=num_actions, history_length=run_config['history_length'] + 1).to(
                device)
        elif run_config['model'] == 'DeepQNetwork':
            Q_net = DeepQNetwork(num_actions=num_actions, history_length=run_config['history_length'] + 1).to(device)
            Q_target_net = DeepQNetwork(num_actions=num_actions, history_length=run_config['history_length'] + 1).to(
                device)
        else:
            raise ValueError('{} not implmented.'.format(run_config['model']))

        agent = DQNAgent(Q=Q_net, Q_target=Q_target_net, num_actions=num_actions, **run_config)
        agent.load(os.path.join(run_path, 'agent.pt'))

        episode_rewards = []
        for i in range(n_test_episodes):
            stats = run_episode(env, agent, deterministic=True, history_length=run_config['history_length'],
                                do_training=False, rendering=True, normalize_images=run_config['normalize_images'],
                                skip_frames=run_config['skip_frames'], max_timesteps=1000)
            episode_rewards.append(stats.episode_reward)

        # save results in a dictionary and write them into a .json file
        results = dict()
        results["episode_rewards"] = episode_rewards
        results["mean"] = np.array(episode_rewards).mean()
        results["std"] = np.array(episode_rewards).std()
        fname = "{}/carracing_results_dqn-{}.json".format(run_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
        fh = open(fname, "w")
        json.dump(results, fh)
        fh.close()

        env.close()
        print('... finished')


if __name__ == "__main__":
    main()
