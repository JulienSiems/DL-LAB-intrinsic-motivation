# export DISPLAY=:0

import sys

sys.path.append("../")

from src.agent.dqn_agent import DQNAgent
from src.agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork, InverseModel, ForwardModel, Encoder
from src.agent.intrinsic_reward import IntrinsicRewardGenerator
from src.training import train_online

import retro

# import gym
# from nes_py.wrappers import JoypadSpace
# import gym_super_mario_bros

from utils.utils import *
import click
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('-ne', '--num_episodes', default=10000, type=click.INT, help='train for ... episodes')
@click.option('-ec', '--eval_cycle', default=50, type=click.INT, help='evaluate every ... episodes')
@click.option('-ne', '--num_eval_episodes', default=1, type=click.INT, help='evaluate this many epochs')
@click.option('-K', '--number_replays', default=1, type=click.INT)
@click.option('-bs', '--batch_size', default=32, type=click.INT)
@click.option('-lr', '--learning_rate', default=1e-4, type=click.FLOAT)
@click.option('-ca', '--capacity', default=20000, type=click.INT)
@click.option('-g', '--gamma', default=0.95, type=click.FLOAT)
@click.option('-e', '--epsilon', default=0.1, type=click.FLOAT)
@click.option('-t', '--tau', default=0.01, type=click.FLOAT)
@click.option('-su', '--soft_update', default=False, type=click.BOOL)
@click.option('-hl', '--history_length', default=4, type=click.INT)
@click.option('-sf', '--skip_frames', default=3, type=click.INT)
@click.option('-lf', '--loss_function', default='L2', type=click.Choice(['L1', 'L2']))
@click.option('-al', '--algorithm', default='DDQN', type=click.Choice(['DQN', 'DDQN']))
@click.option('-mo', '--model', default='DeepQNetwork', type=click.Choice(['Resnet', 'Lenet', 'DeepQNetwork']))
@click.option('-su', '--render_training', default=False, type=click.BOOL)
@click.option('-mt', '--max_timesteps', default=5000, type=click.INT)
@click.option('-ni', '--normalize_images', default=True, type=click.BOOL)
@click.option('-nu', '--non_uniform_sampling', default=False, type=click.BOOL)
@click.option('-ms', '--multi_step', default=True, type=click.BOOL)
@click.option('-mss', '--multi_step_size', default=3, type=click.INT)
@click.option('-mu', '--mu_intrinsic', default=10, type=click.INT)
@click.option('-beta', '--beta_intrinsic', default=0.2, type=click.FLOAT)
@click.option('-lambda', '--lambda_intrinsic', default=0.1, type=click.FLOAT)
@click.option('-i', '--intrinsic', default=True, type=click.BOOL)
@click.option('-e', '--extrinsic', default=False, type=click.BOOL)
@click.option('-uq', '--update_q_target', default=1000, type=click.INT,
              help='How many steps to pass between each q_target update')
@click.option('-es', '--epsilon_schedule', default=True, type=click.BOOL)
@click.option('-est', '--epsilon_start', default=0.9, type=click.FLOAT)
@click.option('-end', '--epsilon_end', default=0.05, type=click.FLOAT)
@click.option('-edc', '--epsilon_decay', default=30000, type=click.INT)
@click.option('-vdy', '--virtual_display', default=False, type=click.BOOL)
@click.option('-s', '--seed', default=0, type=click.INT)
def main(num_episodes, eval_cycle, num_eval_episodes, number_replays, batch_size, learning_rate, capacity, gamma,
         epsilon, tau, soft_update, history_length, skip_frames, loss_function, algorithm, model, render_training,
         max_timesteps, normalize_images, non_uniform_sampling, multi_step, multi_step_size, mu_intrinsic,
         beta_intrinsic, lambda_intrinsic, intrinsic, extrinsic, update_q_target, epsilon_schedule, epsilon_start,
         epsilon_end, epsilon_decay, virtual_display, seed):
    # Set seed
    torch.manual_seed(seed)
    # Create experiment directory with run configuration
    writer = setup_experiment_folder_writer(inspect.currentframe(), name='Mario', log_dir='Breakout2',
                                            args_for_filename=['algorithm', 'loss_function', 'num_episodes',
                                                               'number_replays'])

    if virtual_display:
        if render_training:
            print('On the tfpool computers this will probably not work together. Better deactivate render training when using the virtual display.')
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(224, 240))
        display.start()

    # env = retro.make(game='MontezumaRevenge-Atari2600')
    # env = retro.make(game='SpaceInvaders-Atari2600')
    # env = retro.make(game='BreakOut-Atari2600',  use_restricted_actions=retro.Actions.DISCRETE)

    env = retro.make(game='SuperMarioBros-Nes', use_restricted_actions=retro.Actions.DISCRETE)
    # env = gym_super_mario_bros.make('SuperMarioBros-v0').unwrapped

    num_actions = env.action_space.n

    state_dim = (history_length + 1, 84, 84)

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

    # Intrinsic reward networks
    state_encoder = Encoder(history_length=history_length + 1).to(device)
    inverse_dynamics_model = InverseModel(num_actions=num_actions).to(device)
    forward_dynamics_model = ForwardModel(num_actions=num_actions).to(device)

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
                     update_q_target=update_q_target, lambda_intrinsic=lambda_intrinsic, intrinsic=intrinsic,
                     epsilon_start=epsilon_start, epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                     extrinsic=extrinsic)

    train_online(env=env, agent=agent, writer=writer, num_episodes=num_episodes, eval_cycle=eval_cycle,
                 num_eval_episodes=num_eval_episodes, soft_update=soft_update, skip_frames=skip_frames,
                 history_length=history_length, rendering=render_training, max_timesteps=max_timesteps,
                 normalize_images=normalize_images)
    writer.close()


if __name__ == "__main__":
    main()
