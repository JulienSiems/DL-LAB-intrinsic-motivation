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


def run_episode(env, agent, deterministic, skip_frames=0, do_training=True, rendering=False, max_timesteps=1000,
                history_length=1):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    # plt.interactive(True)
    stats = EpisodeStats()

    # Save history
    history_buffer = []

    step = 0
    e_state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    # env.viewer.window.dispatch_events()

    # append image history to first state
    state = np.zeros((1, e_state['image'].shape[0], e_state['image'].shape[1]))
    # print(env.agent_pos)
    state[0, env.agent_pos[0], env.agent_pos[1]] = 1
    # plt.imshow(state[0, ...], vmin=0, vmax=1)
    # n_state = state_preprocessing(state)
    # plt.imshow(n_state[0,0,...], cmap='gray', vmin=0, vmax=1)
    history_buffer.extend([state] * history_length)
    h_state = np.array(history_buffer)
    # state = np.expand_dims(np.squeeze(np.squeeze(h_state, axis=2), axis=1), axis=0)
    state = h_state

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

            next_state = np.zeros((1, e_next_state['image'].shape[0], e_next_state['image'].shape[1]))
            next_state[0, env.agent_pos[0], env.agent_pos[1]] = 1

            # next_state = state_preprocessing(next_state)
            history_buffer.pop()
            history_buffer.insert(0, next_state)
            h_state = np.array(history_buffer)
            # next_state = np.expand_dims(np.squeeze(np.squeeze(h_state, axis=2), axis=1), axis=0)
            next_state = h_state

            if do_training:
                visitation_map[env.agent_pos[0], env.agent_pos[1]] += 1

            intrinsic_reward = agent.get_intrinsic_reward(state, next_state, action)
            if agent.use_extrinsic_reward:
                r += intrinsic_reward
            else:
                r = intrinsic_reward
            reward += r

            if rendering:
                env.render('human')

            if terminal:
                break

        print(step, action_id, env.agent_pos, reward)

        if do_training:
            # if (step * (skip_frames + 1)) > max_timesteps:
            #     terminal = True
            loss = agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id, loss)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=1, skip_frames=0, model_dir="./models_gridworld",
                 tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = SummaryWriter(log_dir=os.path.join(tensorboard_dir, "train"),
                                filename_suffix="-Carracing_dqn_hist{}_b{}_vc{}.pt".format(
                                    history_length, agent.batch_size, eval_cycle))

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
                           'max_eval_reward': 0, 'Qnet_final': 0, 'Qtargetnet_final': 0, 'optimizer_final': 0}
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

        print("Episode: {} Epsilon: {}	Timesteps: {}	Reward: {}".format(i, agent.epsilon, max_timesteps,
                                                                              stats.episode_reward))

        # evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            sns.heatmap(np.transpose(visitation_map), annot=True, linewidths=.5)
            plt.savefig('visit_map_'+str(i)+'.png', bbox_inches="tight")
            plt.clf()
            eval_rewards = []
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True,
                                    history_length=history_length, max_timesteps=val_max_time_step)
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
            agent.save(checkpoint_data, os.path.join(model_dir, "dqn_agent_hist{}_b{}_vc{}.pt".format(
                history_length, agent.batch_size, eval_cycle)))

    tensorboard.close()


def state_preprocessing(state):
    return np.expand_dims(rgb2gray(np.expand_dims(state, axis=0)), axis=1)


if __name__ == "__main__":

    # env = gym.make('CarRacing-v0').unwrapped
    env = gym.make('MiniGrid-Empty-16x16-v0')
    # env = gym.MiniGridEnv(grid=16, agent_view_size=7)

    history_length = 1
    skip_frames = 0
    state_dim = (history_length, 16, 16)
    num_actions = 3
    # action_distribution = [0.3, 0.15, 0.15, 0.35, 0.05]
    action_distribution = None
    replay_buffer_size = 3e4
    use_extrinsic_reward = False

    visitation_map = np.zeros((16, 16))

    if torch.cuda.is_available():
        batch_size = 32
        num_episodes = 2005
        num_eval_episodes = 5
        eval_cycle = 50
        val_max_time_step = 1000
        debug_flag = False
    else:
        batch_size = 16
        num_episodes = 1005
        num_eval_episodes = 1
        eval_cycle = 10
        val_max_time_step = 50
        debug_flag = False

    # Define Q network, target network and DQN agent
    qnet = ICMModel(in_shape=state_dim, n_classes=num_actions)
    qtarget = ICMModel(in_shape=state_dim, n_classes=num_actions)
    agent = DQNAgent(Q=qnet, Q_target=qtarget, num_actions=num_actions, gamma=0.95, batch_size=batch_size, epsilon=0.1,
                     tau=0.001, lr=1e-3, state_dim=state_dim, do_training=True,
                     replay_buffer_size=replay_buffer_size, act_dist=action_distribution, use_extrinsic_reward=use_extrinsic_reward)

    train_online(env, agent, num_episodes=num_episodes, history_length=history_length, skip_frames=skip_frames,
                 model_dir="./models_gridworld")

