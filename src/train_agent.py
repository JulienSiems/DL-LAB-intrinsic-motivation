# export DISPLAY=:0

import sys

sys.path.append("../")

import torch
import torch.nn
from src.agent.dqn_agent import DQNAgent
from src.agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork, InverseModel, ForwardModel, Encoder
from src.agent.intrinsic_reward import IntrinsicRewardGenerator
from src.training import train_online

from utils.utils import *
import click

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

envs = ['VizDoom', 'Mario', 'GridWorld', 'Pong', 'CartPole', 'CarRacing']
maps = {
    envs[0]: ['my_way_home_org', 'my_way_home_spwnhard', 'my_way_home_spwnhard_nogoal']
}


@click.command()
@click.option('-ne', '--num_episodes', default=10000, type=click.INT, help='train for ... episodes')
@click.option('-ec', '--eval_cycle', default=100, type=click.INT, help='evaluate every ... episodes')
@click.option('-nee', '--num_eval_episodes', default=1, type=click.INT, help='evaluate this many epochs')
@click.option('-tens', '--train_every_n_steps', default=4, type=click.INT)
@click.option('-tnt', '--train_n_times', default=1, type=click.INT)
@click.option('-bs', '--batch_size', default=32, type=click.INT)
@click.option('-lr', '--learning_rate', default=1e-3, type=click.FLOAT)
@click.option('-ac', '--activation', default='ReLU', type=click.Choice(['ReLU', 'ELU', 'LeakyReLU']))
@click.option('-ca', '--capacity', default=2 ** 19, type=click.INT)
@click.option('-g', '--gamma', default=0.9999, type=click.FLOAT)
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
@click.option('-mt', '--max_timesteps', default=1000, type=click.INT)
@click.option('-ni', '--normalize_images', default=True, type=click.BOOL)
@click.option('-nu', '--non_uniform_sampling', default=False, type=click.BOOL)
@click.option('-nsr', '--n_step_reward', default=3, type=click.INT)
@click.option('-mu', '--mu_intrinsic', default=0.001, type=click.FLOAT)
@click.option('-beta', '--beta_intrinsic', default=0.2, type=click.FLOAT)
@click.option('-lambda', '--lambda_intrinsic', default=0.1, type=click.FLOAT)
@click.option('-in', '--intrinsic', default=True, type=click.BOOL)
@click.option('-res', '--residual_icm_forward', default=False, type=click.BOOL)
@click.option('-ih', '--use_history_in_icm', default=True, type=click.BOOL)
@click.option('-ex', '--extrinsic', default=True, type=click.BOOL)
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
@click.option('-per_bdc', '--prio_er_beta_decay', default=1000000, type=click.INT)
@click.option('-ip', '--init_prio', default=500.0, type=click.FLOAT)
@click.option('-fe', '--fixed_encoder', default=False, type=click.BOOL)
@click.option('-du', '--duelling', default=False, type=click.BOOL)
@click.option('-iqn', '--iqn', default=False, type=click.BOOL)
@click.option('-iqn_n', '--iqn_n', default=64, type=click.INT)
@click.option('-iqn_np', '--iqn_np', default=64, type=click.INT)
@click.option('-iqn_k', '--iqn_k', default=128, type=click.INT)
@click.option('-iqn_ted', '--iqn_tau_embed_dim', default=64, type=click.INT)
@click.option('-iqn_dmt', '--iqn_det_max_train', default=False, type=click.BOOL)
@click.option('-iqn_dma', '--iqn_det_max_act', default=False, type=click.BOOL)
@click.option('-hk', '--huber_kappa', default=1.0, type=click.FLOAT)
@click.option('-sh', '--state_height', default=42, type=click.INT)
@click.option('-sw', '--state_width', default=42, type=click.INT)
@click.option('-nmf', '--number_model_files', default=10, type=click.INT)
@click.option('-sct', '--simple_coverage_threshold', default=10, type=click.INT,
              help='Number of separate actions that are required in each segment to be counted as a visited segment.')
@click.option('-gcg', '--geometric_coverage_gamma', default=.99, type=click.FLOAT)
def main(num_episodes, eval_cycle, num_eval_episodes, train_every_n_steps, train_n_times, batch_size, learning_rate,
         capacity, gamma, epsilon, tau, soft_update, history_length, skip_frames, ddqn, model, environment, map,
         activation, render_training, max_timesteps, normalize_images, non_uniform_sampling, n_step_reward,
         mu_intrinsic, beta_intrinsic, lambda_intrinsic, intrinsic, residual_icm_forward, use_history_in_icm, extrinsic,
         update_q_target, epsilon_schedule, epsilon_start, epsilon_end, epsilon_decay, virtual_display, seed,
         pre_intrinsic, experience_replay, prio_er_alpha, prio_er_beta_start, prio_er_beta_end, prio_er_beta_decay,
         init_prio, fixed_encoder, duelling, iqn, iqn_n, iqn_np, iqn_k, iqn_tau_embed_dim, iqn_det_max_train,
         iqn_det_max_act, huber_kappa, state_height, state_width, number_model_files, simple_coverage_threshold,
         geometric_coverage_gamma):
    # Set seed
    torch.manual_seed(seed)
    # Create experiment directory with run configuration
    args_for_filename = ['environment', 'map', 'extrinsic', 'intrinsic', 'fixed_encoder', 'ddqn', 'duelling', 'iqn',
                         'experience_replay', 'soft_update', 'n_step_reward']
    if environment == envs[0]:
        from vizdoom_env.vizdoom_env import DoomEnv
        env = DoomEnv(map_name=map, render=render_training)
        writer = setup_experiment_folder_writer(inspect.currentframe(), name='Vizdoom', log_dir='vizdoom',
                                                args_for_filename=args_for_filename)
        # placeholder for non uniform action probabilities. change to something sensible if wanted.
        nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
        num_actions = env.action_space.n
    else:
        if virtual_display:
            if render_training:
                print(
                    """On the tfpool computers this will probably not work together.
                    Better deactivate render_training when using the virtual display."""
                )
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
                                                    args_for_filename=args_for_filename)
            nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
            num_actions = env.action_space.n
        elif environment == envs[2]:
            import gym_minigrid
            from src.train_gridworld import ClassicalGridworldWrapper
            grid_size = 100
            env = gym_minigrid.envs.EmptyEnv(size=grid_size)
            env = ClassicalGridworldWrapper(env)
            writer = setup_experiment_folder_writer(inspect.currentframe(), name='GridWorld', log_dir='gridworld',
                                                    args_for_filename=args_for_filename)
            nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
            num_actions = env.action_space.n
        elif environment == envs[3]:
            import gym
            env = gym.make('Pong-v0')
            writer = setup_experiment_folder_writer(inspect.currentframe(), name='Pong', log_dir='pong',
                                                    args_for_filename=args_for_filename)
            nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
            num_actions = env.action_space.n
        elif environment == envs[4]:
            import gym
            env = gym.make("CartPole-v0").unwrapped
            writer = setup_experiment_folder_writer(inspect.currentframe(), name='CartPole', log_dir='cartpole',
                                                    args_for_filename=args_for_filename)
            nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
            num_actions = env.action_space.n
        elif environment == envs[5]:
            import gym
            env = gym.make('CarRacing-v0').unwrapped
            writer = setup_experiment_folder_writer(inspect.currentframe(), name='CarRacing', log_dir='carracing',
                                                    args_for_filename=args_for_filename)
            nu_action_probs = [0.45, 0.15, 0.15, 0.15, 0.1]
            num_actions = 5
        else:
            raise NotImplementedError()

    channels = 1  # greyscale images
    state_dim = (channels, state_height, state_width)  # not taking history_length into account. handled later.

    # Define Q network, target network and DQN agent
    if model == 'Resnet':
        CNN = ResnetVariant
    elif model == 'Lenet':
        CNN = LeNetVariant
    elif model == 'DeepQNetwork':
        CNN = DeepQNetwork
    else:
        raise ValueError('{} not implemented'.format(model))

    activation = {'ReLU': torch.nn.ReLU, 'ELU': torch.nn.ELU, 'LeakyReLU': torch.nn.LeakyReLU}[activation]

    Q_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length,
                duelling=duelling, iqn=iqn, activation=activation, embedding_dim=iqn_tau_embed_dim).to(device)
    Q_target_net = CNN(in_dim=state_dim, num_actions=num_actions, history_length=history_length,
                       duelling=duelling, iqn=iqn, activation=activation, embedding_dim=iqn_tau_embed_dim).to(device)

    state_encoder = Encoder(in_dim=state_dim, history_length=history_length,
                            use_history=use_history_in_icm).to(device)
    # push a dummy input through state_encoder to get output dimension which is needed to build dynamics models.
    tmp_inp = torch.zeros(size=(1, channels * (history_length if use_history_in_icm else 1), state_height, state_width))
    tmp_out = state_encoder(tmp_inp.to(device))
    inverse_dynamics_model = InverseModel(num_actions=num_actions, input_dim=2 * tmp_out.shape[1]).to(device)
    forward_dynamics_model = ForwardModel(num_actions=num_actions, state_dim=tmp_out.shape[1]).to(device)

    intrinsic_reward_network = IntrinsicRewardGenerator(state_encoder=state_encoder,
                                                        inverse_dynamics_model=inverse_dynamics_model,
                                                        forward_dynamics_model=forward_dynamics_model,
                                                        num_actions=num_actions,
                                                        fixed_encoder=fixed_encoder,
                                                        residual_forward=residual_icm_forward,
                                                        use_history=use_history_in_icm)

    agent = DQNAgent(Q=Q_net, Q_target=Q_target_net, intrinsic_reward_generator=intrinsic_reward_network,
                     num_actions=num_actions, gamma=gamma, batch_size=batch_size, tau=tau, epsilon=epsilon,
                     capacity=capacity, train_every_n_steps=train_every_n_steps, history_length=history_length,
                     soft_update=soft_update, ddqn=ddqn, n_step_reward=n_step_reward, train_n_times=train_n_times,
                     non_uniform_sampling=non_uniform_sampling, epsilon_schedule=epsilon_schedule, mu=mu_intrinsic,
                     beta=beta_intrinsic, update_q_target=update_q_target, lambda_intrinsic=lambda_intrinsic,
                     intrinsic=intrinsic, epsilon_start=epsilon_start, epsilon_end=epsilon_end, lr=learning_rate,
                     epsilon_decay=epsilon_decay, extrinsic=extrinsic, pre_intrinsic=pre_intrinsic,
                     experience_replay=experience_replay, prio_er_alpha=prio_er_alpha, huber_kappa=huber_kappa,
                     prio_er_beta_start=prio_er_beta_start, prio_er_beta_end=prio_er_beta_end, init_prio=init_prio,
                     prio_er_beta_decay=prio_er_beta_decay, state_dim=state_dim, iqn=iqn, iqn_n=iqn_n, iqn_np=iqn_np,
                     iqn_k=iqn_k, iqn_det_max_train=iqn_det_max_train, iqn_det_max_act=iqn_det_max_act,
                     nu_action_probs=nu_action_probs)

    train_online(env=env, agent=agent, writer=writer, num_episodes=num_episodes, eval_cycle=eval_cycle,
                 num_eval_episodes=num_eval_episodes, soft_update=soft_update, skip_frames=skip_frames,
                 history_length=history_length, rendering=render_training, max_timesteps=max_timesteps,
                 normalize_images=normalize_images, state_dim=state_dim, init_prio=init_prio,
                 num_model_files=number_model_files, simple_coverage_threshold=simple_coverage_threshold,
                 geometric_coverage_gamma=geometric_coverage_gamma)
    writer.close()


if __name__ == "__main__":
    main()
