import sys

sys.path.append("../")

import torch
import numpy as np
import gym
import gym_minigrid
from agent.dqn_agent import DQNAgent
from agent.networks import *
import itertools as it
# from utils.utils import *
from tensorboardX import SummaryWriter
from gym import wrappers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import seaborn as sns
from scipy.stats import wasserstein_distance
from datetime import datetime
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from train_gridworld import ClassicalGridworldWrapper

grid_size = 16
env = gym.make('MiniGrid-Empty-16x16-v0')
num_actions = 3

env = ClassicalGridworldWrapper(env)
num_actions = 4

history_length = 1
skip_frames = 0
state_dim = (history_length, grid_size, grid_size)
action_distribution = None
replay_buffer_size = 3e4
use_icm = True
use_extrinsic_reward = False
agent_policy = 'e_greedy'
icm_eta = 0.2

visitation_map = np.zeros((grid_size, grid_size))

if torch.cuda.is_available():
    batch_size = 32
    num_episodes = 2005
    num_eval_episodes = 5
    eval_cycle = 50
    val_max_time_step = 1000
    debug_flag = False
else:
    batch_size = 16
    num_episodes = 51
    num_eval_episodes = 1
    eval_cycle = 10
    val_max_time_step = 50
    debug_flag = False

qnet = ICMModel(in_shape=state_dim, n_classes=num_actions)
qtarget = ICMModel(in_shape=state_dim, n_classes=num_actions)
agent = DQNAgent(Q=qnet, Q_target=qtarget, num_actions=num_actions, gamma=0.95, batch_size=batch_size, epsilon=0.1,
                 tau=0.001, lr=1e-3, state_dim=state_dim, do_training=True,
                 replay_buffer_size=replay_buffer_size, act_dist=action_distribution, use_icm=use_icm,
                 use_extrinsic_reward=use_extrinsic_reward, policy=agent_policy, icm_eta=icm_eta)

root_dir = './no_env_Graphs'
data = {}
file_name = "dqn_agent.pt"
for dirname, dirnames, filenames in os.walk(root_dir):
    df = 0
    df_exist = False
    for subdirname in dirnames:
        print(os.path.join(dirname, subdirname))

        if dirname != root_dir:
            checkpoint_data = agent.load(os.path.join(dirname, subdirname, file_name))

            if not df_exist:
                df = pd.DataFrame(checkpoint_data, columns=['eval_epoch', 'exploration_coverage', 'exploration_dist', 'exploration_max_freq', 'exploration_goal_reached'])
                data[dirname] = df
                df_exist = True
            else:
                tempdf = pd.DataFrame(checkpoint_data, columns=['eval_epoch', 'exploration_coverage', 'exploration_dist', 'exploration_max_freq', 'exploration_goal_reached'])
                data[dirname] = data[dirname].append(tempdf)

sns.set()
f, axes = plt.subplots(1, 2)
for dir, dflist in data.items():
    df = pd.DataFrame(dflist)

    sns.lineplot(x='eval_epoch', y='exploration_coverage', data=df, label=dir, ci=None, ax=axes[0])
    sns.lineplot(x='eval_epoch', y='exploration_dist', data=df, ci=None, ax=axes[1])

plt.show()
# plt.savefig('Exploration Metric.png')

