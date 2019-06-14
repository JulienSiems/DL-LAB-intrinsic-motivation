# export DISPLAY=:0 

import sys

sys.path.append("../")

import gym
from src.agent.dqn_agent import DQNAgent
from src.agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork

from utils.utils import *
import click
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_episode(env, agent, deterministic, history_length, skip_frames, max_timesteps, normalize_images,
                do_training=True, rendering=False, soft_update=False):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    stats = EpisodeStats()
    # Erase the n_step buffer
    agent.nstep_buffer = []

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state, normalize=normalize_images)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(history_length + 1, 96, 96)
    # state = state.reshape(3, 96, 96) / 255.0
    while True:
        # get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(history_length + 1, 96, 96)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps or stats.episode_reward < -20:
            if agent.multi_step:
                agent.finish_nstep()
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
    bw = rgb2gray(state).reshape(96, 96)
    if normalize:
        bw = bw / 255.0
    return bw


@click.command()
@click.option('-ne', '--num_episodes', default=1000, type=click.INT, help='train for ... episodes')
@click.option('-ec', '--eval_cycle', default=50, type=click.INT, help='evaluate every ... episodes')
@click.option('-ne', '--num_eval_episodes', default=1, type=click.INT, help='evaluate this many epochs')
@click.option('-K', '--number_replays', default=1, type=click.INT)
@click.option('-bs', '--batch_size', default=32, type=click.INT)
@click.option('-lr', '--learning_rate', default=1e-4, type=click.FLOAT)
@click.option('-ca', '--capacity', default=10000, type=click.INT)
@click.option('-g', '--gamma', default=0.95, type=click.FLOAT)
@click.option('-e', '--epsilon', default=0.1, type=click.FLOAT)
@click.option('-t', '--tau', default=0.01, type=click.FLOAT)
@click.option('-su', '--soft_update', default=False, type=click.BOOL)
@click.option('-hl', '--history_length', default=4, type=click.INT)
@click.option('-sf', '--skip_frames', default=3, type=click.INT)
@click.option('-lf', '--loss_function', default='L1', type=click.Choice(['L1', 'L2']))
@click.option('-al', '--algorithm', default='DDQN', type=click.Choice(['DQN', 'DDQN']))
@click.option('-mo', '--model', default='DeepQNetwork', type=click.Choice(['Resnet', 'Lenet', 'DeepQNetwork']))
@click.option('-su', '--render_training', default=False, type=click.BOOL)
@click.option('-mt', '--max_timesteps', default=1000, type=click.INT)
@click.option('-ni', '--normalize_images', default=True, type=click.BOOL)
@click.option('-nu', '--non_uniform_sampling', default=True, type=click.BOOL)
@click.option('-es', '--epsilon_schedule', default=True, type=click.BOOL)
@click.option('-ms', '--multi_step', default=True, type=click.BOOL)
@click.option('-mss', '--multi_step_size', default=3, type=click.INT)
@click.option('-s', '--seed', default=0, type=click.INT)
def main(num_episodes, eval_cycle, num_eval_episodes, number_replays, batch_size, learning_rate, capacity, gamma,
         epsilon, tau, soft_update, history_length, skip_frames, loss_function, algorithm, model, render_training,
         max_timesteps, normalize_images, non_uniform_sampling, epsilon_schedule, multi_step, multi_step_size, seed):
    # Set seed
    torch.manual_seed(seed)
    # Create experiment directory with run configuration
    writer = setup_experiment_folder_writer(inspect.currentframe(), name='car', log_dir='carracing_report_2',
                                            args_for_filename=['algorithm', 'loss_function', 'num_episodes',
                                                               'number_replays'])

    # launch stuff inside
    # virtual display here
    env = gym.make('CarRacing-v0').unwrapped

    num_actions = 5

    # Define Q network, target network and DQN agent
    if model == 'Resnet':
        CNN = ResnetVariant
    elif model == 'Lenet':
        CNN = LeNetVariant
    elif model == 'DeepQNetwork':
        CNN = DeepQNetwork
    else:
        raise ValueError('{} not implemented'.format(model))

    Q_net = CNN(num_actions=num_actions, history_length=history_length + 1).to(device)
    Q_target_net = CNN(num_actions=num_actions, history_length=history_length + 1).to(device)

    agent = DQNAgent(Q=Q_net, Q_target=Q_target_net, num_actions=num_actions, gamma=gamma, batch_size=batch_size,
                     tau=tau, epsilon=epsilon, lr=learning_rate, capacity=capacity, number_replays=number_replays,
                     loss_function=loss_function, soft_update=soft_update, algorithm=algorithm, multi_step=multi_step,
                     multi_step_size=multi_step_size, non_uniform_sampling=non_uniform_sampling,
                     epsilon_schedule=epsilon_schedule)

    train_online(env=env, agent=agent, writer=writer, num_episodes=num_episodes, eval_cycle=eval_cycle,
                 num_eval_episodes=num_eval_episodes, soft_update=soft_update, skip_frames=skip_frames,
                 history_length=history_length, rendering=render_training, max_timesteps=max_timesteps,
                 normalize_images=normalize_images)
    writer.close()


if __name__ == "__main__":
    main()
