# export DISPLAY=:0 

import sys

sys.path.append("../")

import torch
import gym
from src.agent.dqn_agent import DQNAgent
from src.agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork, InverseModel, ForwardModel, Encoder
from src.agent.intrinsic_reward import IntrinsicRewardGenerator

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

class ClassicalGridworldWrapper(gym.Wrapper):
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


def run_episode(env, agent, deterministic, history_length, skip_frames, max_timesteps, normalize_images,
                do_training=True, rendering=False, soft_update=False):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    global visitation_map

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()
    state = np.zeros((1, env.width, env.height))
    state[0, env.agent_pos[0], env.agent_pos[1]] = 1

    # fix bug of corrupted states without rendering in gym environment
    # env.viewer.window.dispatch_events()

    # append image history to first state
    # state = state_preprocessing(state, normalize=normalize_images)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape([history_length + 1, env.height, env.width])
    while True:
        # get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state=state, deterministic=deterministic)
        # action = np.zeros((1, agent.num_actions))
        # action[0, action_id] = 1.0
        # action = torch.nn.functional.one_hot(action_id.long(), num_classes=agent.num_actions)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action_id)
            terminal = False

            next_state = np.zeros((1, env.width, env.height))
            next_state[0, env.agent_pos[0], env.agent_pos[1]] = 1

            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        # next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape([history_length + 1, env.height, env.width])

        if do_training:
            visitation_map[env.agent_pos[0], env.agent_pos[1]] += 1
            agent.append_to_replay(state=state, action=action_id, next_state=next_state, reward=reward,
                                   terminal=terminal)
            agent.train()

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps or stats.episode_reward < -20:
            if agent.multi_step:
                # Finish n step buffer
                agent.finish_n_step()
            break
        step += 1

    print('epsilon_threshold', agent.eps_threshold)

    # Update the target network
    if not soft_update:
        agent.Q_target.load_state_dict(agent.Q.state_dict())

    return stats


def train_online(env, agent, writer, num_episodes, eval_cycle, num_eval_episodes, soft_update, skip_frames,
                 history_length, rendering, max_timesteps, normalize_images):
    print("... train agent")

    global visitation_map
    visitation_map = np.zeros((env.height, env.width))
    uniform_prob = np.array(visitation_map)
    uniform_prob[:, :] = 1 / ((visitation_map.shape[0] - 2) * (visitation_map.shape[1] - 2))
    uniform_prob[:, 0] = 0
    uniform_prob[:, -1] = 0
    uniform_prob[0, :] = 0
    uniform_prob[-1, :] = 0

    for i in range(num_episodes):
        print("episode %d" % i)
        max_timesteps_current = max_timesteps
        stats = run_episode(env, agent, max_timesteps=max_timesteps_current, deterministic=False, do_training=True,
                            rendering=rendering, soft_update=soft_update, skip_frames=skip_frames,
                            history_length=history_length, normalize_images=normalize_images)

        writer.add_scalar('train_episode_reward', stats.episode_reward, global_step=i)
        writer.add_scalar('train_straight', stats.get_action_usage(STRAIGHT), global_step=i)
        writer.add_scalar('train_left', stats.get_action_usage(LEFT), global_step=i)
        writer.add_scalar('train_right', stats.get_action_usage(RIGHT), global_step=i)
        writer.add_scalar('train_accel', stats.get_action_usage(ACCELERATE), global_step=i)
        writer.add_scalar('train_brake', stats.get_action_usage(BRAKE), global_step=i)

        # EVALUATION
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            total_visits = sum(sum(visitation_map))
            obs_prob = visitation_map / total_visits
            dist, _, _ = cv2.EMD(arr_to_sig(obs_prob), arr_to_sig(uniform_prob), cv2.DIST_L2)
            coverage = np.count_nonzero(visitation_map) / (
                        (visitation_map.shape[0] - 2) * (visitation_map.shape[1] - 2))
            writer.add_scalar('exploration_coverage', coverage, global_step=i)
            writer.add_scalar('exploration_EM_distance', dist, global_step=i)

            plt.figure(figsize=(20, 20))
            sns.heatmap(np.transpose(visitation_map), annot=True, linewidths=.5, square=True)
            plt.title('EM Distance ' + str(dist))
            plt.savefig(os.path.join(writer.logdir, 'visit_map_' + str(i) + '.png'), bbox_inches="tight")
            plt.clf()
            print('exploration coverage: {}     dist: {}'.format(coverage, dist))

            stats = []
            for j in range(num_eval_episodes):
                stats.append(run_episode(env, agent, deterministic=True, do_training=False, max_timesteps=1000,
                                         history_length=history_length, skip_frames=skip_frames,
                                         normalize_images=normalize_images))
            stats_agg = [stat.episode_reward for stat in stats]
            episode_reward_mean, episode_reward_std = np.mean(stats_agg), np.std(stats_agg)
            print('Validation {} +- {}'.format(episode_reward_mean, episode_reward_std))
            print('Replay buffer length', len(agent.replay_buffer._data))
            writer.add_scalar('val_episode_reward_mean', episode_reward_mean, global_step=i)
            writer.add_scalar('val_episode_reward_std', episode_reward_std, global_step=i)

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            model_dir = agent.save(os.path.join(writer.logdir, "agent.pt"))
            print("Model saved in file: %s" % model_dir)


def state_preprocessing(state, normalize=True):
    image_resized = Image.fromarray(state).resize((42, 42), Image.ANTIALIAS)
    image_resized_bw = rgb2gray(np.array(image_resized))
    if normalize:
        image_resized_bw = image_resized_bw / 255.0
    return image_resized_bw


@click.command()
@click.option('-ne', '--num_episodes', default=51, type=click.INT, help='train for ... episodes')
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
@click.option('-lf', '--loss_function', default='L1', type=click.Choice(['L1', 'L2']))
@click.option('-al', '--algorithm', default='DDQN', type=click.Choice(['DQN', 'DDQN']))
@click.option('-mo', '--model', default='DeepQNetwork', type=click.Choice(['Resnet', 'Lenet', 'DeepQNetwork']))
@click.option('-su', '--render_training', default=True, type=click.BOOL)
@click.option('-mt', '--max_timesteps', default=100, type=click.INT)
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
def main(num_episodes, eval_cycle, num_eval_episodes, number_replays, batch_size, learning_rate, capacity, gamma,
         epsilon, tau, soft_update, history_length, skip_frames, loss_function, algorithm, model, render_training,
         max_timesteps, normalize_images, non_uniform_sampling, epsilon_schedule, multi_step, multi_step_size,
         mu_intrinsic, beta_intrinsic, lambda_intrinsic, intrinsic, extrinsic, seed):
    # Set seed
    torch.manual_seed(seed)
    # Create experiment directory with run configuration
    writer = setup_experiment_folder_writer(inspect.currentframe(), name='gridworld', log_dir='classic_gridworld_report',
                                            args_for_filename=['algorithm', 'loss_function', 'num_episodes',
                                                               'number_replays'])

    # launch stuff inside
    # virtual display here
    grid_size = 16
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

    dummy_input = torch.zeros(1, state_dim[0], state_dim[1], state_dim[2])
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

    train_online(env=env, agent=agent, writer=writer, num_episodes=num_episodes, eval_cycle=eval_cycle,
                 num_eval_episodes=num_eval_episodes, soft_update=soft_update, skip_frames=skip_frames,
                 history_length=history_length, rendering=render_training, max_timesteps=max_timesteps,
                 normalize_images=normalize_images)
    writer.close()


if __name__ == "__main__":
    main()
