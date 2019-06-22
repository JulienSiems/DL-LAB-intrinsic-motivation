# export DISPLAY=:0 

import sys

sys.path.append("../")

import torch
import numpy as np
import gym
import gym_minigrid
from agent.dqn_agent import DQNAgent
from agent.networks import *
import itertools as it
from utils.utils import *
from tensorboardX import SummaryWriter
from gym import wrappers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import seaborn as sns
from scipy.stats import wasserstein_distance
from datetime import datetime
import cv2
from enum import IntEnum
import random
from src.agent.networks import DeepQNetwork, InverseModel, ForwardModel, Encoder
from src.agent.intrinsic_reward import IntrinsicRewardGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arr_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""

    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i, j], i, j])
            count += 1
    return sig


# AGENT_DIR_TO_STR = {
#             0: '>',
#             1: 'V',
#             2: '<',
#             3: '^'
#         }

class ClassicalGridworldWrapper(gym.Wrapper): #gym_minigrid.MiniGridEnv
    class ClassicActions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3

    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2

    class Directions(IntEnum):
        right = 0
        down = 1
        left = 2
        up = 3

    DICT_OFFSET = 5
    ACTION_CONVERSION_MAP = {
        (ClassicActions.left + DICT_OFFSET * Directions.right): [Actions.right, Actions.right, Actions.forward],
        (ClassicActions.left + DICT_OFFSET * Directions.down): [Actions.right, Actions.forward],
        (ClassicActions.left + DICT_OFFSET * Directions.left): [Actions.forward],
        (ClassicActions.left + DICT_OFFSET * Directions.up): [Actions.left, Actions.forward],

        (ClassicActions.right + DICT_OFFSET * Directions.right): [Actions.forward],
        (ClassicActions.right + DICT_OFFSET * Directions.down): [Actions.left, Actions.forward],
        (ClassicActions.right + DICT_OFFSET * Directions.left): [Actions.right, Actions.right, Actions.forward],
        (ClassicActions.right + DICT_OFFSET * Directions.up): [Actions.right, Actions.forward],

        (ClassicActions.up + DICT_OFFSET * Directions.right): [Actions.left, Actions.forward],
        (ClassicActions.up + DICT_OFFSET * Directions.down): [Actions.right, Actions.right, Actions.forward],
        (ClassicActions.up + DICT_OFFSET * Directions.left): [Actions.right, Actions.forward],
        (ClassicActions.up + DICT_OFFSET * Directions.up): [Actions.forward],

        (ClassicActions.down + DICT_OFFSET * Directions.right): [Actions.right, Actions.forward],
        (ClassicActions.down + DICT_OFFSET * Directions.down): [Actions.forward],
        (ClassicActions.down + DICT_OFFSET * Directions.left): [Actions.left, Actions.forward],
        (ClassicActions.down + DICT_OFFSET * Directions.up): [Actions.right, Actions.right, Actions.forward],
    }

    def __init__(self, env=None):
        super(ClassicalGridworldWrapper, self).__init__(env)

    def step(self, action):
        reward = 0
        next_state = 0
        terminal = 0
        action_sequence = ClassicalGridworldWrapper.ACTION_CONVERSION_MAP[action + ClassicalGridworldWrapper.DICT_OFFSET * self.agent_dir]
        for act in action_sequence:
            next_state, r, terminal, info = self.env.step(act)
            reward += r
            if terminal:
                break
        return next_state, r, terminal, info

    def reset(self):
        return self.env.reset()


def run_episode(env, agent, deterministic, skip_frames=0, do_training=True, rendering=False, max_timesteps=1000,
                history_length=1):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    global visitation_map

    stats = EpisodeStats()

    # Save history
    history_buffer = []

    step = 0
    e_state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    # env.viewer.window.dispatch_events()

    # append image history to first state
    state = np.zeros((1, env.width, env.height))
    # print(env.agent_pos)
    state[0, env.agent_pos[0], env.agent_pos[1]] = 1
    # plt.imshow(state[0, ...], vmin=0, vmax=1)
    # n_state = state_preprocessing(state)
    # plt.imshow(n_state[0,0,...], cmap='gray', vmin=0, vmax=1)
    history_buffer.extend([state] * history_length)
    h_state = np.array(history_buffer)
    # state = np.expand_dims(np.squeeze(np.squeeze(h_state, axis=2), axis=1), axis=0)
    state = h_state
    state = np.array(state).reshape([history_length, env.height, env.width])

    visitation_map[env.agent_pos[0], env.agent_pos[1]] += 1

    loss = 0
    while True:
        # get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state=state, deterministic=deterministic)
        # action = id_to_action(action_id, max_speed=0.8)
        action = np.zeros((1, agent.num_actions))
        action[0, action_id] = 1.0

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            e_next_state, r, terminal, info = env.step(action_id)
            r = 0
            terminal = False

            next_state = np.zeros((1, env.width, env.height))
            next_state[0, env.agent_pos[0], env.agent_pos[1]] = 1

            # next_state = state_preprocessing(next_state)
            history_buffer.pop()
            history_buffer.insert(0, next_state)
            h_state = np.array(history_buffer)
            # next_state = np.expand_dims(np.squeeze(np.squeeze(h_state, axis=2), axis=1), axis=0)
            next_state = h_state
            next_state = np.array(next_state).reshape([history_length, env.height, env.width])

            if do_training:
                visitation_map[env.agent_pos[0], env.agent_pos[1]] += 1

            # if agent.use_icm and do_training:
            #     intrinsic_reward = agent.get_intrinsic_reward(state, next_state, action)
            #     if agent.use_extrinsic_reward:
            #         r += intrinsic_reward
            #     else:
            #         r = intrinsic_reward
            reward += r

            if rendering:
                env.render('human')

            if terminal:
                break

        # print(step, action_id, env.agent_pos, reward)

        if do_training:
            # if (step * (skip_frames + 1)) > max_timesteps:
            #     terminal = True
            loss = agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id, loss)
        print(step, reward, action_id, sum(sum(visitation_map)))

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(env,
                 agent, num_episodes, history_length=1, skip_frames=0, model_dir="./models_gridworld",
                 tensorboard_dir="./tensorboard"):
    global visitation_map
    visitation_map = np.zeros((env.height, env.width))
    uniform_prob = np.array(visitation_map)
    uniform_prob[:, :] = 1 / ((visitation_map.shape[0] - 2) * (visitation_map.shape[1] - 2))
    uniform_prob[:, 0] = 0
    uniform_prob[:, -1] = 0
    uniform_prob[0, :] = 0
    uniform_prob[-1, :] = 0

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    save_dir = os.path.join(model_dir, 'Gridworld_dqn_hist{}_b{}_vc{}_icm{}_eta{}_p_{}_{}'.format(
                                    history_length, agent.batch_size, eval_cycle, int(agent.use_icm), agent.icm_eta,
                                    agent.policy, datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.mkdir(save_dir)

    print("... train agent")
    tensorboard = SummaryWriter(logdir=save_dir, filename_suffix="-Gridworld_dqn")

    # if debug_flag:
    #     dummy_input = torch.zeros(1, 1, 16, 16)
    #     if torch.cuda.is_available():
    #         dummy_input = dummy_input.to('cuda')
    #     p = agent.Q(state=dummy_input, next_state=None, action=None, mode=ICM_GET_ONLY_Q_OUT)
    #     tensorboard.add_graph(agent.Q, dummy_input)

    try:
        checkpoint_data = agent.load(os.path.join(model_dir, "dqn_agent_hist{}_b{}_vc{}.pt".format(history_length,
                                                                                                     agent.batch_size,
                                                                                                     eval_cycle)))
        for id, i in enumerate(checkpoint_data['epoch']):
            tensorboard.add_scalar("Train/Episode Reward", checkpoint_data['tr_reward'][id], i)
            tensorboard.add_scalar("Train/straight", checkpoint_data['tr_a_straight'][id], i)
            tensorboard.add_scalar("Train/left", checkpoint_data['tr_a_left'][id], i)
            tensorboard.add_scalar("Train/right", checkpoint_data['tr_a_right'][id], i)
            tensorboard.add_scalar("Train/accel", checkpoint_data['tr_a_accel'][id], i)
            tensorboard.add_scalar("Train/brake", checkpoint_data['tr_a_brake'][id], i)
        epoch_start = id + 1
        max_reward = checkpoint_data['max_eval_reward']
        for id, i in enumerate(checkpoint_data['eval_epoch']):
            tensorboard.add_scalar("Evaluation/Mean Eval Reward", checkpoint_data['eval_reward'][id], i)

        print("=> loading checkpoint success".format(model_dir))
    except:
        print("=> loading checkpoint failed".format(model_dir))
        checkpoint_data = {'epoch': [], 'tr_reward': [], 'tr_a_straight': [], 'tr_a_left': [], 'tr_a_right': [],
                           'tr_a_accel': [], 'tr_a_brake': [],
                           'eval_epoch': [], 'eval_reward': [], 'Qnet': 0, 'Qtargetnet': 0, 'optimizer': 0,
                           'max_eval_reward': 0, 'Qnet_final': 0, 'Qtargetnet_final': 0, 'optimizer_final': 0,
                           'exploration_dist': [], 'exploration_goal_reached': [], 'exploration_max_freq': [], 'exploration_coverage': []
                           }
        epoch_start = 0
        max_reward = -20000

    agent.set_eval_mode()
    for i in range(epoch_start, num_episodes):
        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        # max_timesteps = int(min(pow(i / (num_episodes - 100), 1.5) * 1000 + 200, 1000))
        max_timesteps = 100
        stats = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=False, do_training=True,
                            rendering=True, history_length=history_length, skip_frames=skip_frames)
        # agent.epsilon_max = max(agent.epsilon_final, agent.epsilon_start - i / (num_episodes / 10))
        # agent.epsilon = agent.epsilon_max

        tensorboard.add_scalar("Train/Episode Reward", stats.episode_reward, i + 1)
        tensorboard.add_scalar("Train/straight", stats.get_action_usage(STRAIGHT), i + 1)
        tensorboard.add_scalar("Train/left", stats.get_action_usage(LEFT), i + 1)
        tensorboard.add_scalar("Train/right", stats.get_action_usage(RIGHT), i + 1)
        tensorboard.add_scalar("Train/accel", stats.get_action_usage(ACCELERATE), i + 1)
        tensorboard.add_scalar("Train/brake", stats.get_action_usage(BRAKE), i + 1)

        checkpoint_data['epoch'].append(i + 1)
        checkpoint_data['tr_reward'].append(stats.episode_reward)
        checkpoint_data['tr_a_straight'].append(stats.get_action_usage(STRAIGHT))
        checkpoint_data['tr_a_left'].append(stats.get_action_usage(LEFT))
        checkpoint_data['tr_a_right'].append(stats.get_action_usage(RIGHT))
        checkpoint_data['tr_a_accel'].append(stats.get_action_usage(ACCELERATE))
        checkpoint_data['tr_a_brake'].append(stats.get_action_usage(BRAKE))

        print("Episode: {} Epsilon: {}	    Timesteps: {}	Reward: {}".format(i, agent.epsilon, max_timesteps,
                                                                              stats.episode_reward))

        # evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            total_visits = sum(sum(visitation_map))
            # uniform_prob = uniform_prob.ravel()
            obs_prob = visitation_map / total_visits
            # wdist = wasserstein_distance(obs_prob.ravel(), uniform_prob.ravel())
            # w1dist, _, _ = cv2.EMD(arr_to_sig(obs_prob.reshape(1, 16*16)), arr_to_sig(uniform_prob.reshape(1, 16*16)), cv2.DIST_L2)
            dist, _, _ = cv2.EMD(arr_to_sig(obs_prob), arr_to_sig(uniform_prob), cv2.DIST_L2)
            coverage = np.count_nonzero(visitation_map) / ((visitation_map.shape[0]-2) * (visitation_map.shape[1]-2))
            checkpoint_data['exploration_dist'].append(dist)
            checkpoint_data['exploration_goal_reached'].append(visitation_map[-2, -2])
            checkpoint_data['exploration_max_freq'].append(visitation_map.max())
            checkpoint_data['exploration_coverage'].append(coverage)
            plt.figure(figsize=(20, 20))
            sns.heatmap(np.transpose(visitation_map), annot=True, linewidths=.5, square=True)

            plt.title('EM Distance '+str(dist))
            plt.savefig(os.path.join(save_dir, 'visit_map_'+str(i)+'.png'), bbox_inches="tight")
            plt.clf()
            print('exploration coverage: {}     dist: {}'.format(coverage, dist))

            eval_rewards = []
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True,
                                    history_length=history_length, max_timesteps=5)
                eval_rewards.append(stats.episode_reward)
            mean_reward = sum(eval_rewards) / num_eval_episodes
            tensorboard.add_scalar("Evaluation/Mean Eval Reward", mean_reward, i + 1)
            checkpoint_data['eval_epoch'].append(i + 1)
            checkpoint_data['eval_reward'].append(mean_reward)
            print("Mean Eval Rewards: {}".format(mean_reward))
            print("Mean Eval Rewards: {}    Max Reward: {}".format(mean_reward, max_reward))

            if max_reward <= mean_reward:
                max_reward = mean_reward
                checkpoint_data['Qnet'] = agent.Q.state_dict()
                checkpoint_data['Qtargetnet'] = agent.Q_target.state_dict()
                checkpoint_data['optimizer'] = agent.optimizer.state_dict()
                checkpoint_data['max_eval_reward'] = max_reward
            checkpoint_data['Qnet_final'] = agent.Q.state_dict()
            checkpoint_data['Qtargetnet_final'] = agent.Q_target.state_dict()
            checkpoint_data['optimizer_final'] = agent.optimizer.state_dict()
            agent.save(checkpoint_data, os.path.join(save_dir, "dqn_agent.pt"))

    tensorboard.close()


def state_preprocessing(state):
    return np.expand_dims(rgb2gray(np.expand_dims(state, axis=0)), axis=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    grid_size = 16
    env = gym_minigrid.envs.EmptyEnv(size=grid_size)
    num_actions = 3

    env = ClassicalGridworldWrapper(env)
    num_actions = 4

    history_length = 1
    skip_frames = 0
    state_dim = (history_length, grid_size, grid_size)
    action_distribution = None
    replay_buffer_size = 3e4
    use_icm = False
    use_extrinsic_reward = False
    agent_policy = 'e_greedy'
    icm_eta = 5

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
        val_max_time_step = 100
        debug_flag = False

    # Define Q network, target network and DQN agent
    # qnet = ICMModel(in_shape=state_dim, n_classes=num_actions)
    # qtarget = ICMModel(in_shape=state_dim, n_classes=num_actions)
    CNN = DeepQNetwork
    Q_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length).to(device)
    Q_target_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length).to(device)

    state_encoder = Encoder(history_length=history_length).to(device)
    # Intrinsic reward networks

    dummy_input = torch.zeros(1, state_dim[0], state_dim[1], state_dim[2]).to(device)
    out_cnn = state_encoder(dummy_input)
    out_cnn = out_cnn.view(out_cnn.size(0), -1)
    cnn_out_size = out_cnn.shape[1]

    inverse_dynamics_model = InverseModel(num_actions=num_actions, input_dimension=cnn_out_size * 2).to(device)
    forward_dynamics_model = ForwardModel(num_actions=num_actions, dim_s=cnn_out_size,
                                          output_dimension=cnn_out_size).to(device)

    intrinsic_reward_network = IntrinsicRewardGenerator(state_encoder=state_encoder,
                                                        inverse_dynamics_model=inverse_dynamics_model,
                                                        forward_dynamics_model=forward_dynamics_model,
                                                        num_actions=num_actions)


    agent = DQNAgent(Q=Q_net, Q_target=Q_target_net, intrinsic_reward_generator=intrinsic_reward_network, num_actions=num_actions, gamma=0.95, batch_size=batch_size, epsilon=0.1,
                     tau=0.01, lr=1e-3, state_dim=state_dim, do_training=True,
                     replay_buffer_size=replay_buffer_size, act_dist=action_distribution, use_icm=use_icm,
                     use_extrinsic_reward=use_extrinsic_reward, policy=agent_policy, icm_eta=icm_eta)

    for _ in range(1):
        train_online(env, agent, num_episodes=num_episodes, history_length=history_length, skip_frames=skip_frames,
                 model_dir="./models_gridworld")

