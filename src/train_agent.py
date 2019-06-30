# export DISPLAY=:0

import sys

sys.path.append("../")

import torch
from src.agent.dqn_agent import DQNAgent
from src.agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork, InverseModel, ForwardModel, Encoder
from src.agent.intrinsic_reward import IntrinsicRewardGenerator
from src.training import train_online

from utils.utils import *
import click

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

envs = ['VizDoom', 'Mario', 'GridWorld']
maps = {
    envs[0]: ['my_way_home_spwnhard', 'my_way_home_org', 'my_way_home_spwnhard_nogoal']
}


@click.command()
@click.option('-ne', '--num_episodes', default=10000, type=click.INT, help='train for ... episodes')
@click.option('-ec', '--eval_cycle', default=50, type=click.INT, help='evaluate every ... episodes')
@click.option('-nee', '--num_eval_episodes', default=1, type=click.INT, help='evaluate this many epochs')
@click.option('-K', '--number_replays', default=1, type=click.INT)
@click.option('-bs', '--batch_size', default=32, type=click.INT)
@click.option('-lr', '--learning_rate', default=1e-3, type=click.FLOAT)
@click.option('-ca', '--capacity', default=2**17, type=click.INT)
@click.option('-g', '--gamma', default=0.95, type=click.FLOAT)
@click.option('-e', '--epsilon', default=0.1, type=click.FLOAT)
@click.option('-t', '--tau', default=0.01, type=click.FLOAT)
@click.option('-su', '--soft_update', default=False, type=click.BOOL)
@click.option('-hl', '--history_length', default=4, type=click.INT)
@click.option('-sf', '--skip_frames', default=3, type=click.INT)
@click.option('-ddqn', '--ddqn', default=False, type=click.BOOL)
@click.option('-mo', '--model', default='DeepQNetwork', type=click.Choice(['Resnet', 'Lenet', 'DeepQNetwork']))
@click.option('-env', '--environment', default=envs[0], type=click.Choice(envs))
@click.option('-mp', '--map', default=maps[envs[0]][0], type=click.Choice(maps[envs[0]]))
@click.option('-rt', '--render_training', default=False, type=click.BOOL)
@click.option('-mt', '--max_timesteps', default=2100, type=click.INT)
@click.option('-ni', '--normalize_images', default=True, type=click.BOOL)
@click.option('-nu', '--non_uniform_sampling', default=False, type=click.BOOL)
@click.option('-ms', '--multi_step', default=True, type=click.BOOL)
@click.option('-mss', '--multi_step_size', default=3, type=click.INT)
@click.option('-mu', '--mu_intrinsic', default=10, type=click.INT)
@click.option('-beta', '--beta_intrinsic', default=0.2, type=click.FLOAT)
@click.option('-lambda', '--lambda_intrinsic', default=0.1, type=click.FLOAT)
@click.option('-in', '--intrinsic', default=True, type=click.BOOL)
@click.option('-res', '--residual_icm_forward', default=False, type=click.BOOL)
@click.option('-ih', '--use_history_in_icm', default=True, type=click.BOOL)
@click.option('-ex', '--extrinsic', default=False, type=click.BOOL)
@click.option('-uq', '--update_q_target', default=10000, type=click.INT,
              help='How many steps to pass between each q_target update')
@click.option('-es', '--epsilon_schedule', default=False, type=click.BOOL)
@click.option('-est', '--epsilon_start', default=0.9, type=click.FLOAT)
@click.option('-end', '--epsilon_end', default=0.05, type=click.FLOAT)
@click.option('-edc', '--epsilon_decay', default=30000, type=click.INT)
@click.option('-vdy', '--virtual_display', default=False, type=click.BOOL)
@click.option('-s', '--seed', default=0, type=click.INT)
@click.option('-pre_icm', '--pre_intrinsic', default=False, type=click.BOOL)
@click.option('-er', '--experience_replay', default='Uniform', type=click.Choice(['Uniform', 'Prioritized']))
@click.option('-per_a', '--prio_er_alpha', default=0.6, type=click.FLOAT)
@click.option('-per_bs', '--prio_er_beta_start', default=0.4, type=click.FLOAT)
@click.option('-per_be', '--prio_er_beta_end', default=1.0, type=click.FLOAT)
@click.option('-per_bdc', '--prio_er_beta_decay', default=250000, type=click.INT)
@click.option('-fe', '--fixed_encoder', default=False, type=click.BOOL)
@click.option('-du', '--duelling', default=False, type=click.BOOL)
@click.option('-iqn', '--iqn', default=False, type=click.BOOL)
@click.option('-iqn_n', '--iqn_n', default=32, type=click.INT)
@click.option('-iqn_np', '--iqn_np', default=32, type=click.INT)
@click.option('-iqn_k', '--iqn_k', default=32, type=click.INT)
@click.option('-iqn_ted', '--iqn_tau_embed_dim', default=64, type=click.INT)
@click.option('-hk', '--huber_kappa', default=1.0, type=click.FLOAT)
@click.option('-sh', '--state_height', default=42, type=click.INT)
@click.option('-sw', '--state_width', default=42, type=click.INT)
def main(num_episodes, eval_cycle, num_eval_episodes, number_replays, batch_size, learning_rate, capacity, gamma,
         epsilon, tau, soft_update, history_length, skip_frames, ddqn, model, environment, map,
         render_training, max_timesteps, normalize_images, non_uniform_sampling, multi_step, multi_step_size,
         mu_intrinsic, beta_intrinsic, lambda_intrinsic, intrinsic, residual_icm_forward, use_history_in_icm, extrinsic,
         update_q_target, epsilon_schedule,
         epsilon_start, epsilon_end, epsilon_decay, virtual_display, seed, pre_intrinsic, experience_replay,
         prio_er_alpha, prio_er_beta_start, prio_er_beta_end, prio_er_beta_decay, fixed_encoder, duelling, iqn, iqn_n,
         iqn_np, iqn_k, iqn_tau_embed_dim, huber_kappa, state_height, state_width):
    # Set seed
    torch.manual_seed(seed)
    # Create experiment directory with run configuration
    if environment == envs[0]:
        from vizdoom_env.vizdoom_env import DoomEnv
        env = DoomEnv(map_name=map, render=render_training)
        writer = setup_experiment_folder_writer(inspect.currentframe(), name='Vizdoom', log_dir='vizdoom',
                                                args_for_filename=['environment', 'extrinsic', 'intrinsic',
                                                                   'fixed_encoder', 'ddqn', 'duelling', 'iqn',
                                                                   'experience_replay', 'soft_update'])
    else:
        if virtual_display:
            if render_training:
                print(
                    'On the tfpool computers this will probably not work together. Better deactivate render training when using the virtual display.')
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(224, 240))
            display.start()
        if environment == envs[1]:
            from nes_py.wrappers import JoypadSpace
            import gym_super_mario_bros
            from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
            # env = retro.make(game='SuperMarioBros-Nes')
            env = gym_super_mario_bros.make('SuperMarioBros-v0').unwrapped
            env = JoypadSpace(env, COMPLEX_MOVEMENT)
            writer = setup_experiment_folder_writer(inspect.currentframe(), name='Mario', log_dir='mario',
                                                    args_for_filename=['environment', 'extrinsic', 'intrinsic',
                                                                       'fixed_encoder', 'ddqn', 'duelling', 'iqn',
                                                                       'experience_replay', 'soft_update'])
        elif environment == envs[2]:
            import gym_minigrid
            from src.train_gridworld import ClassicalGridworldWrapper
            grid_size = 100
            env = gym_minigrid.envs.EmptyEnv(size=grid_size)
            env = ClassicalGridworldWrapper(env)
            writer = setup_experiment_folder_writer(inspect.currentframe(), name='GridWorld', log_dir='gridworld',
                                                    args_for_filename=['environment', 'extrinsic', 'intrinsic',
                                                                       'fixed_encoder', 'ddqn', 'duelling', 'iqn',
                                                                       'experience_replay', 'soft_update'])
        else:
            raise NotImplementedError()

    num_actions = env.action_space.n

    state_dim = (history_length, state_height, state_width)

    # Define Q network, target network and DQN agent
    if model == 'Resnet':
        CNN = ResnetVariant
    elif model == 'Lenet':
        CNN = LeNetVariant
    elif model == 'DeepQNetwork':
        CNN = DeepQNetwork
    else:
        raise ValueError('{} not implemented'.format(model))

    Q_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length,
                duelling=duelling, iqn=iqn, embedding_dim=iqn_tau_embed_dim).to(device)
    Q_target_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length,
                       duelling=duelling, iqn=iqn, embedding_dim=iqn_tau_embed_dim).to(device)

    # Intrinsic reward networks
    if use_history_in_icm:
        encoder_in_channels = history_length
    else:
        encoder_in_channels = 1

    state_encoder = Encoder(history_length=encoder_in_channels).to(device)
    inverse_dynamics_model = InverseModel(num_actions=num_actions).to(device)
    forward_dynamics_model = ForwardModel(num_actions=num_actions).to(device)

    intrinsic_reward_network = IntrinsicRewardGenerator(state_encoder=state_encoder,
                                                        inverse_dynamics_model=inverse_dynamics_model,
                                                        forward_dynamics_model=forward_dynamics_model,
                                                        num_actions=num_actions,
                                                        fixed_encoder=fixed_encoder,
                                                        residual_forward=residual_icm_forward,
                                                        use_history=use_history_in_icm)

    agent = DQNAgent(Q=Q_net, Q_target=Q_target_net, intrinsic_reward_generator=intrinsic_reward_network,
                     num_actions=num_actions, gamma=gamma, batch_size=batch_size, tau=tau, epsilon=epsilon,
                     lr=learning_rate, capacity=capacity, number_replays=number_replays, soft_update=soft_update,
                     ddqn=ddqn, multi_step=multi_step, multi_step_size=multi_step_size,
                     non_uniform_sampling=non_uniform_sampling, epsilon_schedule=epsilon_schedule, mu=mu_intrinsic,
                     beta=beta_intrinsic, update_q_target=update_q_target, lambda_intrinsic=lambda_intrinsic,
                     intrinsic=intrinsic, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                     epsilon_decay=epsilon_decay, extrinsic=extrinsic, pre_intrinsic=pre_intrinsic,
                     experience_replay=experience_replay, prio_er_alpha=prio_er_alpha,
                     prio_er_beta_start=prio_er_beta_start, prio_er_beta_end=prio_er_beta_end,
                     prio_er_beta_decay=prio_er_beta_decay, state_dim=state_dim, iqn=iqn, iqn_n=iqn_n, iqn_np=iqn_np,
                     iqn_k=iqn_k, huber_kappa=huber_kappa)

    train_online(env=env, agent=agent, writer=writer, num_episodes=num_episodes, eval_cycle=eval_cycle,
                 num_eval_episodes=num_eval_episodes, soft_update=soft_update, skip_frames=skip_frames,
                 history_length=history_length, rendering=render_training, max_timesteps=max_timesteps,
                 normalize_images=normalize_images, state_dim=state_dim)
    writer.close()


if __name__ == "__main__":
    main()
