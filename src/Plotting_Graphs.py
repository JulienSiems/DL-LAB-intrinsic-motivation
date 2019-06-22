import sys

sys.path.append("../")

import torch
import gym
from src.agent.dqn_agent import DQNAgent
from src.agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork, InverseModel, ForwardModel, Encoder
from src.agent.intrinsic_reward import IntrinsicRewardGenerator

import tensorflow as tf
from utils.utils import *
import click
from PIL import Image

import gym_minigrid
import cv2
from enum import IntEnum
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from gym import wrappers
import pandas as pd

from train_gridworld import ClassicalGridworldWrapper


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_size = 100
env = gym_minigrid.envs.EmptyEnv(size=grid_size)
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


@click.command()
@click.option('-ne', '--num_episodes', default=1001, type=click.INT, help='train for ... episodes')
@click.option('-ec', '--eval_cycle', default=10, type=click.INT, help='evaluate every ... episodes')
@click.option('-ne', '--num_eval_episodes', default=1, type=click.INT, help='evaluate this many epochs')
@click.option('-K', '--number_replays', default=1, type=click.INT)
@click.option('-bs', '--batch_size', default=16, type=click.INT)
@click.option('-lr', '--learning_rate', default=1e-3, type=click.FLOAT)
@click.option('-ca', '--capacity', default=30000, type=click.INT)
@click.option('-g', '--gamma', default=0.95, type=click.FLOAT)
@click.option('-e', '--epsilon', default=0.1, type=click.FLOAT)
@click.option('-t', '--tau', default=0.01, type=click.FLOAT)
@click.option('-su', '--soft_update', default=True, type=click.BOOL)
@click.option('-hl', '--history_length', default=0, type=click.INT)
@click.option('-sf', '--skip_frames', default=0, type=click.INT)
@click.option('-lf', '--loss_function', default='L2', type=click.Choice(['L1', 'L2']))
@click.option('-al', '--algorithm', default='DDQN', type=click.Choice(['DQN', 'DDQN']))
@click.option('-mo', '--model', default='DeepQNetwork', type=click.Choice(['Resnet', 'Lenet', 'DeepQNetwork']))
@click.option('-su', '--render_training', default=True, type=click.BOOL)
@click.option('-mt', '--max_timesteps', default=500, type=click.INT)
@click.option('-ni', '--normalize_images', default=True, type=click.BOOL)
@click.option('-nu', '--non_uniform_sampling', default=False, type=click.BOOL)
@click.option('-es', '--epsilon_schedule', default=False, type=click.BOOL)
@click.option('-ms', '--multi_step', default=False, type=click.BOOL)
@click.option('-mss', '--multi_step_size', default=3, type=click.INT)
@click.option('-mu', '--mu_intrinsic', default=0.2, type=click.FLOAT)
@click.option('-beta', '--beta_intrinsic', default=0.2, type=click.FLOAT)
@click.option('-lambda', '--lambda_intrinsic', default=0.1, type=click.FLOAT)
@click.option('-i', '--intrinsic', default=True, type=click.BOOL)
@click.option('-e', '--extrinsic', default=False, type=click.BOOL)
@click.option('-s', '--seed', default=0, type=click.INT)
@click.option('-grid', '--env_grid', default=100, type=click.INT)
def main(num_episodes, eval_cycle, num_eval_episodes, number_replays, batch_size, learning_rate, capacity, gamma,
         epsilon, tau, soft_update, history_length, skip_frames, loss_function, algorithm, model, render_training,
         max_timesteps, normalize_images, non_uniform_sampling, epsilon_schedule, multi_step, multi_step_size,
         mu_intrinsic, beta_intrinsic, lambda_intrinsic, intrinsic, extrinsic, seed, env_grid):
    # Set seed
    torch.manual_seed(seed)

    # launch stuff inside
    # virtual display here
    grid_size = env_grid
    env = gym_minigrid.envs.EmptyEnv(size=grid_size)
    num_actions = 3

    env = ClassicalGridworldWrapper(env)
    num_actions = 4

    state_dim = (history_length + 1, grid_size, grid_size)

    # Define Q network, target network and DQN agent
    if model == 'Resnet':
        CNN = ResnetVariant
    elif model == 'Lenet':
        CNN = LeNetVariant
    elif model == 'DeepQNetwork':
        CNN = DeepQNetwork
    else:
        raise ValueError('{} not implemented'.format(model))

    Q_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length + 1).to(device)
    Q_target_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length + 1).to(device)

    state_encoder = Encoder(history_length=history_length + 1).to(device)
    # Intrinsic reward networks

    dummy_input = torch.zeros(1, state_dim[0], state_dim[1], state_dim[2]).to(device)
    out_cnn = state_encoder(dummy_input)
    out_cnn = out_cnn.view(out_cnn.size(0), -1)
    cnn_out_size = out_cnn.shape[1]

    inverse_dynamics_model = InverseModel(num_actions=num_actions, input_dimension=cnn_out_size*2).to(device)
    forward_dynamics_model = ForwardModel(num_actions=num_actions, dim_s=cnn_out_size, output_dimension=cnn_out_size).to(device)

    intrinsic_reward_network = IntrinsicRewardGenerator(state_encoder=state_encoder,
                                                        inverse_dynamics_model=inverse_dynamics_model,
                                                        forward_dynamics_model=forward_dynamics_model,
                                                        num_actions=num_actions)

    agent = DQNAgent(Q=Q_net, Q_target=Q_target_net, intrinsic_reward_generator=intrinsic_reward_network,
                     num_actions=num_actions, gamma=gamma, batch_size=batch_size, tau=tau, epsilon=epsilon,
                     lr=learning_rate, capacity=capacity, number_replays=number_replays, loss_function=loss_function,
                     soft_update=soft_update, algorithm=algorithm, multi_step=multi_step,
                     multi_step_size=multi_step_size, non_uniform_sampling=non_uniform_sampling,
                     epsilon_schedule=epsilon_schedule, mu=mu_intrinsic, beta=beta_intrinsic,
                     lambda_intrinsic=lambda_intrinsic, intrinsic=intrinsic, extrinsic=extrinsic)



    root_dir = './100grid'
    data = {}
    file_name = "dqn_agent.pt"
    for dirname, dirnames, filenames in os.walk(root_dir):
        df = 0
        df_exist = False
        for subdirname in dirnames:
            subdir_path = os.path.join(dirname, subdirname)

            if dirname != root_dir:
                print(subdir_path)
                for _, _, filenames in os.walk(subdir_path):
                    # print(filenames)
                    for fname in filenames:
                        if 'events.out.tfevents.' in fname:
                            data_file_name = fname
                            break
                    break
                print(data_file_name)

                data_dict = {'exploration_coverage': [], 'exploration_EM_distance': [], 'epoch_coverage': [], 'epoch_dist': []}
                max_epoch = 0
                for e in tf.train.summary_iterator(os.path.join(dirname, subdirname, data_file_name)):
                    for v in e.summary.value:
                        if v.tag == 'exploration_coverage' or v.tag == 'exploration_EM_distance':
                            data_dict[v.tag].append(v.simple_value)
                            if v.tag == 'exploration_coverage':
                                data_dict['epoch_coverage'].append(e.step)
                            elif v.tag == 'exploration_EM_distance':
                                data_dict['epoch_dist'].append(e.step)

                            if max_epoch < len(data_dict['epoch_coverage']):
                                max_epoch = len(data_dict['epoch_coverage'])
                            if max_epoch < len(data_dict['epoch_dist']):
                                max_epoch = len(data_dict['epoch_dist'])

                for k, v in data_dict.items():
                    while len(v) != max_epoch:
                        v.append(v[-1])

                if not df_exist:
                    df = pd.DataFrame(data_dict, columns=['epoch_coverage', 'epoch_dist', 'exploration_coverage', 'exploration_EM_distance'])
                    data[dirname] = df
                    df_exist = True
                    # print(data[dirname])
                else:
                    tempdf = pd.DataFrame(data_dict, columns=['epoch_coverage', 'epoch_dist', 'exploration_coverage', 'exploration_EM_distance'])
                    data[dirname] = data[dirname].append(tempdf)
                    # print(data[dirname])

    sns.set()
    f, axes = plt.subplots(1, 2)
    for dir, dflist in data.items():
        df = pd.DataFrame(dflist)
        # print(df)
        sns.lineplot(x='epoch_coverage', y='exploration_coverage', data=df, label=dir, ci=None, ax=axes[0])
        sns.lineplot(x='epoch_dist', y='exploration_EM_distance', data=df, ci=None, ax=axes[1])

    plt.show()
    plt.savefig('Exploration Metric.png')


if __name__ == "__main__":
    main()
