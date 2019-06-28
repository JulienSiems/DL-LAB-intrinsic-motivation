import random

import numpy as np
import torch

from src.agent.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, intrinsic_reward_generator, num_actions, capacity, intrinsic, extrinsic,
                 mu, beta, lambda_intrinsic, epsilon_start, epsilon_end, epsilon_decay, update_q_target,
                 experience_replay, prio_er_alpha, prio_er_beta_start, prio_er_beta_end, prio_er_beta_decay, state_dim,
                 iqn, iqn_n, iqn_np, iqn_k,
                 multi_step=True, multi_step_size=3, non_uniform_sampling=False, gamma=0.95, batch_size=64, epsilon=0.1,
                 tau=0.01, lr=1e-4, number_replays=10, loss_function='L1', soft_update=False, algorithm='DQN',
                 epsilon_schedule=False, pre_intrinsic=False, *args, **kwargs):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q
        self.Q_target = Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.Q_target.eval()

        # intrinsic reward generator
        self.intrinsic_reward_generator = intrinsic_reward_generator

        nets = [self.Q,
                self.intrinsic_reward_generator.state_encoder,
                self.intrinsic_reward_generator.inverse_dynamics_model,
                self.intrinsic_reward_generator.forward_dynamics_model]
        parameters = set()
        for net in nets:
            parameters |= set(net.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=lr)

        # define replay buffer
        tmp_state_shape = tuple([1] + list(state_dim))
        self.experience_replay = experience_replay
        self.prio_er_alpha = prio_er_alpha
        self.prio_er_beta_start = prio_er_beta_start
        self.prio_er_beta_end = prio_er_beta_end
        self.prio_er_beta_decay = prio_er_beta_decay
        self.cur_prio_er_beta = prio_er_beta_start
        if experience_replay == 'Uniform':
            self.replay_buffer = UniformReplayBuffer(capacity=capacity, state_shape=tmp_state_shape,
                                                     state_store_dtype=np.float16, state_sample_dtype=np.float32)
        elif experience_replay == 'Prioritized':
            self.replay_buffer = PrioritizedReplayBuffer(capacity=capacity, state_shape=tmp_state_shape,
                                                         state_store_dtype=np.float16, state_sample_dtype=np.float32,
                                                         alpha=prio_er_alpha)
        else:
            raise ValueError('Unknown experience replay buffer type: {}'.format(experience_replay))

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.mu = mu
        self.beta = beta
        self.lambda_intrinsic = lambda_intrinsic
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.pre_intrinsic = pre_intrinsic

        self.number_replays = number_replays
        self.num_actions = num_actions
        self.steps_done = 0
        self.train_steps_done = 0
        self.soft_update = soft_update
        self.non_uniform_sampling = non_uniform_sampling

        self.algorithm = algorithm

        self.epsilon_schedule = epsilon_schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_q_target = update_q_target
        self.n_step_buffer = []
        self.n_steps = multi_step_size
        self.multi_step = multi_step

        self.iqn = iqn
        self.iqn_n = iqn_n
        self.iqn_np = iqn_np
        self.iqn_k = iqn_k

        if loss_function == 'L1':
            self.loss_function = torch.nn.SmoothL1Loss()
        elif loss_function == 'L2':
            self.loss_function = torch.nn.MSELoss()
        else:
            raise ValueError('Loss function {} not implemented.'.format(loss_function))

    # Adapated from https://github.com/qfettes/DeepRL-Tutorials/blob/master/02.NStep_DQN.ipynb
    def append_to_replay(self, state, action, next_state, reward, terminal, beginning, priority):
        if self.multi_step:
            self.n_step_buffer.append((state, action, next_state, reward))

            if len(self.n_step_buffer) < self.n_steps:
                return

            R = sum([self.n_step_buffer[i][3] * (self.gamma ** i) for i in range(self.n_steps)])
            state, action, _, _ = self.n_step_buffer.pop(0)

            self.replay_buffer.add_transition(state, action, R, next_state, terminal, beginning, priority)
        else:
            self.replay_buffer.add_transition(state, action, reward, next_state, terminal, beginning, priority)

    def finish_n_step(self):
        if self.multi_step:
            while len(self.n_step_buffer) > 0:
                R = sum([self.n_step_buffer[i][3] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
                state, action, next_state, _ = self.n_step_buffer.pop(0)
                self.replay_buffer.add_transition(state, action, R, next_state, True, beginning=False, priority=500.0)

    def train(self):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        interpolate_b = min(1.0, self.train_steps_done / self.prio_er_beta_decay)
        self.cur_prio_er_beta = self.prio_er_beta_start * (1.0 - interpolate_b) + self.prio_er_beta_end * interpolate_b

        if self.batch_size > self.replay_buffer.size:
            return None, None, None, None

        for iter in range(self.number_replays):
            # 2. sample next batch and perform batch update: (initially take less than batch_size because of the replay
            # buffer
            batch_state_sequences, batch_actions, batch_rewards, batch_dones, _, weights, sample_idxs = \
                self.replay_buffer.sample_batch(self.batch_size, history_length=1, n_steps=1,
                                                beta=self.cur_prio_er_beta, balance_batch=True)
            batch_states = batch_state_sequences[:, 0, ...]
            batch_next_states = batch_state_sequences[:, 1, ...]
            batch_actions = batch_actions[:, 0]
            batch_rewards = batch_rewards[:, 0]
            batch_dones = batch_dones[:, 0]
            weights = torch.from_numpy(weights).detach().to(device)

            # Set the expected value of a terminated section to zero.
            non_final_mask = torch.from_numpy(np.array(batch_dones != True, dtype=np.uint8))
            non_final_next_states = torch.from_numpy(batch_next_states)[non_final_mask].to(device).float()

            # 2.1 compute td targets and loss
            # td_target =  reward + discount * max_a Q_target(next_state_batch, a)
            next_state_values = torch.zeros(self.batch_size, device=device, dtype=torch.float)
            if 'DQN' == self.algorithm:
                next_state_values[non_final_mask] = torch.max(self.Q_target(non_final_next_states), dim=1)[0]
            elif 'DDQN' == self.algorithm:
                # Double DQN
                # Adapted from https://github.com/Shivanshu-Gupta/Pytorch-Double-DQN/blob/master/agent.py
                next_state_actions = self.Q(non_final_next_states).max(dim=1)[1]
                next_state_values[non_final_mask] = \
                    self.Q_target(non_final_next_states).gather(dim=1, index=next_state_actions.view(-1, 1)).squeeze(1)
            else:
                raise ValueError('Algorithm {} not implemented'.format(self.algorithm))

            if self.intrinsic:
                # Compute intrinsic_reward
                L_I, L_F, intrinsic_reward, l_i = self.intrinsic_reward_generator.compute_intrinsic_reward(
                    state=batch_states,
                    action=batch_actions,
                    next_state=batch_next_states)
                r_i = intrinsic_reward
                intrinsic_reward = intrinsic_reward.detach() * self.mu
            else:
                L_I, L_F, intrinsic_reward, l_i = \
                    [torch.tensor([0], dtype=torch.float, device=device) for _ in range(4)]

            if self.extrinsic or self.pre_intrinsic:
                extrinsic_reward = torch.from_numpy(batch_rewards).to(device).float()
            else:
                extrinsic_reward = 0.0

            reward = extrinsic_reward + (intrinsic_reward if not self.pre_intrinsic else 0.0)

            # Detach from comp graph to avoid that gradients are propagated through the target network.
            next_state_values = next_state_values.detach()
            if self.multi_step:
                td_target = reward + (self.gamma ** self.n_steps) * next_state_values
            else:
                td_target = reward + self.gamma * next_state_values

            # 2.2 update the Q network
            # self.Q.update(...)
            state_action_values = self.Q(torch.from_numpy(batch_states).to(device).float())
            batch_actions_tensor = torch.from_numpy(batch_actions).to(device).view(-1, 1)

            # Choose the action previously taken
            q_pick = torch.gather(state_action_values, dim=1, index=batch_actions_tensor.long())
            q_pick = q_pick.view(-1)
            # Chosen like in this tutorial https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            self.optimizer.zero_grad()
            # print(L_I.item(), L_F.item())
            # td_loss = self.loss_function(input=q_pick, target=td_target.unsqueeze(1))
            td_losses = (q_pick - td_target.detach()).pow(2)

            if self.intrinsic:
                losses = self.lambda_intrinsic * td_losses + (1 - self.beta) * l_i + self.beta * r_i
            else:
                losses = td_losses
            weighted_losses = losses * weights
            weighted_losses.mean().backward()
            self.optimizer.step()

            # update priorities of transitions in replay buffer
            if self.experience_replay == 'Prioritized':
                for s_idx in range(self.batch_size):
                    new_priority = weighted_losses[s_idx] + self.replay_buffer.priority_eps
                    self.replay_buffer.propagate_up(sample_idxs[s_idx], new_priority)

        # call soft update for target network
        if self.soft_update:
            soft_update(self.Q_target, self.Q, self.tau)

        self.train_steps_done += 1
        return losses.mean().item(), td_losses.mean().item(), L_I.item(), L_F.item()

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        sample = random.random()

        if self.epsilon_schedule:
            # Like in pytorch tutorial https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                            np.exp(-1. * self.steps_done / self.epsilon_decay)
        else:
            eps_threshold = self.epsilon
        self.eps_threshold = eps_threshold
        self.steps_done += 1

        r = np.random.uniform()
        if deterministic or sample > eps_threshold:
            with torch.no_grad():
                # take greedy action (argmax)
                state_ = torch.from_numpy(np.expand_dims(state, 0)).to(device).float()
                if self.iqn:
                    # for IQN we have to sample from reward distribution to determine greedy action
                    taus = torch.rand(size=(1, self.iqn_k), dtype=torch.float, device=device)
                    pred = self.Q(state_, taus=taus)
                    action_id = torch.argmax(pred.mean(dim=0)).detach().cpu().numpy()
                else:
                    pred = self.Q(state_)
                    action_id = torch.argmax(pred).detach().cpu().numpy()
        else:
            if self.non_uniform_sampling:
                action_id = \
                    np.random.choice(self.num_actions, 1, p=[0.45, 0.15, 0.15, 0.15, 0.1])[0]
            else:
                action_id = np.random.randint(self.num_actions)
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name, map_location='cpu'))
        self.Q_target.load_state_dict(torch.load(file_name, map_location='cpu'))
