# export DISPLAY=:0

import sys

sys.path.append("../")

import torch
import torch.nn
from src.agent.dqn_agent import DQNAgent
from src.agent.networks import ResnetVariant, LeNetVariant, DeepQNetwork, InverseModel, ForwardModel, Encoder
from src.agent.intrinsic_reward import IntrinsicRewardGenerator
from src.training import eval_offline

from utils.utils import *
import click
import re
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

envs = ['VizDoom', 'Mario', 'GridWorld', 'Pong']
maps = {
    envs[0]: ['my_way_home_org', 'my_way_home_spwnhard', 'my_way_home_spwnhard_nogoal']
}

#
@click.command()
@click.option('-ne', '--num_evals', default=50, type=click.INT)
@click.option('-d', '--dir', default='./vizdoom/eval_runs', type=click.STRING)
@click.option('-a', '--alpha', default=0.1, type=click.FLOAT)
def main(num_evals, dir, alpha):
    def sort_models(fname):
        split_array = re.split('[_ .]', fname)
        return int(split_array[1])
    root_dir = dir
    file_name_list = []
    model_name_list = []
    event_file_list = []
    param_list = []
    for dirname in os.listdir(root_dir):
        # print(dirname)
        subdir_path = os.path.join(root_dir, dirname)
        file_name_list.append(dirname)
        model_list = []
        for filename in os.listdir(subdir_path):
            # print(filename)
            if 'agent_' in filename and '.pt' in filename:
                model_list.append(filename)
            elif 'events.out.tfevents.' in filename:
                event_file_list.append(filename)
            elif 'config.json' in filename:
                param_list.append(filename)
        model_name_list.append(sorted(model_list, key=lambda x: sort_models(x)))

    # print(model_name_list)
    for folder_index in range(len(model_name_list)):
        print(os.path.join(root_dir, file_name_list[folder_index], param_list[folder_index]))

        json_file_name = os.path.join(root_dir, file_name_list[folder_index], param_list[folder_index])
        json_file = open(json_file_name)
        json_str = json_file.read()
        hyperparam = json.loads(json_str)

        num_episodes = hyperparam['num_episodes']
        eval_cycle = hyperparam['eval_cycle']
        num_eval_episodes = hyperparam['num_eval_episodes']
        train_every_n_steps = hyperparam['train_every_n_steps']
        train_n_times = hyperparam['train_n_times']
        batch_size = hyperparam['batch_size']
        learning_rate = hyperparam['learning_rate']
        capacity = hyperparam['capacity']
        gamma = hyperparam['gamma']
        epsilon = hyperparam['epsilon']
        tau = hyperparam['tau']
        soft_update = hyperparam['soft_update']
        history_length = hyperparam['history_length']
        skip_frames = hyperparam['skip_frames']
        ddqn = hyperparam['ddqn']
        model = hyperparam['model']
        environment = hyperparam['environment']
        map = hyperparam['map']
        activation = hyperparam['activation']
        render_training = hyperparam['render_training']
        max_timesteps = hyperparam['max_timesteps']
        normalize_images = hyperparam['normalize_images']
        non_uniform_sampling = hyperparam['non_uniform_sampling']
        n_step_reward = hyperparam['n_step_reward']
        mu_intrinsic = hyperparam['mu_intrinsic']
        beta_intrinsic = hyperparam['beta_intrinsic']
        lambda_intrinsic = hyperparam['lambda_intrinsic']
        intrinsic = hyperparam['intrinsic']
        residual_icm_forward = hyperparam['residual_icm_forward']
        use_history_in_icm = hyperparam['use_history_in_icm']
        extrinsic = hyperparam['extrinsic']
        update_q_target = hyperparam['update_q_target']
        epsilon_schedule = hyperparam['epsilon_schedule']
        epsilon_start = hyperparam['epsilon_start']
        epsilon_end = hyperparam['epsilon_end']
        epsilon_decay = hyperparam['epsilon_decay']
        virtual_display = hyperparam['virtual_display']
        seed = hyperparam['seed']
        pre_intrinsic = hyperparam['pre_intrinsic']
        experience_replay = hyperparam['experience_replay']
        prio_er_alpha = hyperparam['prio_er_alpha']
        prio_er_beta_start = hyperparam['prio_er_beta_start']
        prio_er_beta_end = hyperparam['prio_er_beta_end']
        prio_er_beta_decay = hyperparam['prio_er_beta_decay']
        init_prio = hyperparam['init_prio']
        fixed_encoder = hyperparam['fixed_encoder']
        duelling = hyperparam['duelling']
        iqn = hyperparam['iqn']
        iqn_n = hyperparam['iqn_n']
        iqn_np = hyperparam['iqn_np']
        iqn_k = hyperparam['iqn_k']
        iqn_tau_embed_dim = hyperparam['iqn_tau_embed_dim']
        iqn_det_max_train = hyperparam['iqn_det_max_train']
        iqn_det_max_act = hyperparam['iqn_det_max_act']
        huber_kappa = hyperparam['huber_kappa']
        state_height = hyperparam['state_height']
        state_width = hyperparam['state_width']
        number_model_files = hyperparam['number_model_files']
        simple_coverage_threshold = hyperparam['simple_coverage_threshold']
        geometric_coverage_gamma = hyperparam['geometric_coverage_gamma']
        num_total_steps = hyperparam['num_total_steps']
        store_cycle = hyperparam['store_cycle']
        adam_epsilon = hyperparam['adam_epsilon']
        gradient_clip = hyperparam.get('gradient_clip', False)

        # Set seed
        torch.manual_seed(seed)
        # Create experiment directory with run configuration
        args_for_filename = ['environment', 'map', 'extrinsic', 'intrinsic', 'fixed_encoder', 'ddqn', 'duelling', 'iqn',
                             'experience_replay', 'soft_update', 'n_step_reward']

        if environment == envs[0]:
            from vizdoom_env.vizdoom_env import DoomEnv
            env = DoomEnv(map_name=map, render=render_training)
            writer = setup_experiment_folder_writer(inspect.currentframe(), name='Vizdoom', log_dir='vizdoom_eval',
                                                    args_for_filename=args_for_filename, additional_param=hyperparam)
            # placeholder for non uniform action probabilities. change to something sensible if wanted.
            nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
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
            elif environment == envs[2]:
                import gym_minigrid
                from src.train_gridworld import ClassicalGridworldWrapper
                grid_size = 100
                env = gym_minigrid.envs.EmptyEnv(size=grid_size)
                env = ClassicalGridworldWrapper(env)
                writer = setup_experiment_folder_writer(inspect.currentframe(), name='GridWorld', log_dir='gridworld',
                                                        args_for_filename=args_for_filename)
                nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
            elif environment == envs[3]:
                import gym
                env = gym.make('Pong-v0')
                writer = setup_experiment_folder_writer(inspect.currentframe(), name='Pong', log_dir='pong',
                                                        args_for_filename=args_for_filename)
                nu_action_probs = np.ones(env.action_space.n, dtype=np.float32) / env.action_space.n
            else:
                raise NotImplementedError()

        num_actions = env.action_space.n

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
                         nu_action_probs=nu_action_probs, adam_epsilon=adam_epsilon, gradient_clip=gradient_clip)

        eval_offline(env=env, agent=agent, writer=writer, num_episodes=num_episodes, eval_cycle=eval_cycle,
                     num_eval_episodes=num_eval_episodes, soft_update=soft_update, skip_frames=skip_frames,
                     history_length=history_length, rendering=render_training, max_timesteps=max_timesteps,
                     normalize_images=normalize_images, state_dim=state_dim, init_prio=init_prio,
                     num_model_files=number_model_files, simple_coverage_threshold=simple_coverage_threshold,
                     geometric_coverage_gamma=geometric_coverage_gamma, num_total_steps=num_total_steps,
                     store_cycle=store_cycle, model_name_list=model_name_list[folder_index], alpha=alpha,
                     num_evals=num_evals, path_of_run=os.path.join(root_dir, file_name_list[folder_index]))
        writer.close()


if __name__ == "__main__":
    main()
