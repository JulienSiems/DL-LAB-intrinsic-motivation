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
                 iqn, iqn_n, iqn_np, iqn_k, huber_kappa,
                 multi_step=True, multi_step_size=3, non_uniform_sampling=False, gamma=0.95, batch_size=64, epsilon=0.1,
                 tau=0.01, lr=1e-4, number_replays=10, soft_update=False, ddqn=True, epsilon_schedule=False,
                 pre_intrinsic=False, *args, **kwargs):
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

        self.ddqn = ddqn

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
        self.huber_kappa = huber_kappa

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
            self.optimizer.zero_grad()

            batch_state_sequences, batch_actions, batch_rewards, batch_dones, _, weights, sample_idxs = \
                self.replay_buffer.sample_batch(self.batch_size, history_length=1, n_steps=1,
                                                beta=self.cur_prio_er_beta, balance_batch=True)
            batch_states = batch_state_sequences[:, 0, ...]
            batch_next_states = batch_state_sequences[:, 1, ...]
            batch_actions = batch_actions[:, 0]
            batch_rewards = batch_rewards[:, 0]
            batch_dones = batch_dones[:, 0]
            weights = torch.from_numpy(weights).detach().to(device)

            batch_states_ = torch.from_numpy(batch_states).to(device)
            batch_next_states_ = torch.from_numpy(batch_next_states).to(device)
            batch_actions_ = torch.from_numpy(batch_actions).long().to(device).view(-1, 1)

            # compute mask that weighs values of terminated next states with zero.
            non_final_mask = torch.from_numpy(np.array(batch_dones != True, dtype=np.float32)).to(device)

            if self.iqn:
                # when using IQN, net outputs are of shape (batch_size * num_taus, num_actions)
                taus = torch.rand(size=(self.batch_size, self.iqn_n), dtype=torch.float, device=device)
                taus_prime = torch.rand(size=(self.batch_size, self.iqn_np), dtype=torch.float, device=device)
                taus_tilde = torch.linspace(start=0.0, end=1.0, steps=self.iqn_k + 2, dtype=torch.float, device=device)
                taus_tilde = taus_tilde[1:-1].view(1, self.iqn_k).repeat(self.batch_size, 1)
                # taus_tilde = torch.rand(size=(self.batch_size, self.iqn_k), dtype=torch.float, device=device)
                # predict value of taken action.
                Q_pred = self.Q(batch_states_, taus=taus)
                batch_actions_repeated = batch_actions_.repeat(1, self.iqn_n).view(-1, 1)
                Q_pred_picked = Q_pred.gather(dim=1, index=batch_actions_repeated).squeeze()
                # predict value of best action in next state.
                Q_target_pred_next = self.Q_target(batch_next_states_, taus=taus_prime)
                if self.ddqn:
                    pred_next_to_max = self.Q(batch_next_states_, taus=taus_tilde)
                else:
                    pred_next_to_max = self.Q_target(batch_next_states_, taus=taus_tilde)
                pred_next_to_max = pred_next_to_max.view(self.batch_size, self.iqn_k, self.num_actions)
                max_next_actions = torch.max(pred_next_to_max.mean(dim=1), dim=1)[1]
                max_next_actions_repeated = max_next_actions.view(-1, 1).repeat(1, self.iqn_np).view(-1, 1)
                Q_target_pred_next_picked = Q_target_pred_next.gather(dim=1, index=max_next_actions_repeated).squeeze()
                non_final_mask_repeated = non_final_mask.view(-1, 1).repeat(1, self.iqn_np).flatten()
                Q_target_pred_next_picked = Q_target_pred_next_picked * non_final_mask_repeated
            else:
                # predict value of taken action.
                Q_pred = self.Q(batch_states_)
                Q_pred_picked = Q_pred.gather(dim=1, index=batch_actions_).squeeze()
                # predict value of best action in next state.
                Q_target_pred_next = self.Q_target(batch_next_states_)
                if self.ddqn:
                    pred_next_to_max = self.Q(batch_next_states_)
                else:
                    pred_next_to_max = Q_target_pred_next
                max_next_actions = torch.max(pred_next_to_max, dim=1)[1].view(-1, 1)
                Q_target_pred_next_picked = Q_target_pred_next.gather(dim=1, index=max_next_actions).squeeze()
                Q_target_pred_next_picked = Q_target_pred_next_picked * non_final_mask

            # detach from comp graph to avoid that gradients are propagated through the target network.
            Q_target_pred_next_picked = Q_target_pred_next_picked.detach()

            if self.intrinsic:
                # Compute intrinsic_reward
                L_I, L_F, intrinsic_reward, l_i = self.intrinsic_reward_generator.compute_intrinsic_reward(
                    state=batch_states,
                    action=batch_actions,
                    next_state=batch_next_states)
                # this is not detached because it is used to optimise the forward model.
                r_i = intrinsic_reward
                # this is detached because it's used for the reward to optimise the Q model.
                intrinsic_reward = intrinsic_reward.detach() * self.mu
            else:
                L_I, L_F, intrinsic_reward, l_i = \
                    [torch.tensor([0], dtype=torch.float, device=device) for _ in range(4)]

            if self.extrinsic or self.pre_intrinsic:
                extrinsic_reward = torch.from_numpy(batch_rewards).to(device).float()
            else:
                extrinsic_reward = 0.0

            reward = extrinsic_reward + (intrinsic_reward if not self.pre_intrinsic else 0.0)
            if self.iqn:
                reward = reward.view(-1, 1).repeat(1, self.iqn_np).view(-1, 1).squeeze()

            if self.multi_step:
                td_target = reward + (self.gamma ** self.n_steps) * Q_target_pred_next_picked
            else:
                td_target = reward + self.gamma * Q_target_pred_next_picked

            if self.iqn:
                Q_pred_picked_repeated = Q_pred_picked.view(-1, 1).repeat(1, self.iqn_np).view(-1, 1).squeeze()
                td_target_repeated = \
                    td_target.view(self.batch_size, self.iqn_np).repeat(1, self.iqn_n).view(-1, 1).squeeze()
                taus_repeated = taus.view(-1, 1).repeat(1, self.iqn_np).view(-1, 1).squeeze()
                # difference between predicted quantile values and target samples.
                delta = td_target_repeated - Q_pred_picked_repeated
                # if delta is negative, target is on the left of predicted quantile value.
                ind_left = (delta < 0.0).float()
                # weigh samples left of prediction with 1.0 - tau and the right with tau.
                # if samples are distributed correctly, sides will cancel out to zero.
                side_weight = torch.abs(taus_repeated - ind_left)
                # calculate huber loss of delta
                # case 0: delta < kappa; case 1: delta >= kappa
                abs_delta = torch.abs(delta)
                abs_smaller_kappa_mask = (abs_delta < self.huber_kappa).float()
                huber_case0 = 0.5 * delta.pow(2) * abs_smaller_kappa_mask
                huber_case1 = (self.huber_kappa * (abs_delta - 0.5 * self.huber_kappa) *
                               torch.abs(1.0 - abs_smaller_kappa_mask))
                huber_loss = huber_case0 + huber_case1
                td_losses = huber_loss.view(-1, self.iqn_np).mean(dim=1).view(self.batch_size, self.iqn_n).sum(dim=1)
            else:
                # squared error
                td_losses = (Q_pred_picked - td_target.detach()).pow(2)

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
                    # we evaluate q net at grid points of a linspace ranging from 0.0 to 1.0.
                    taus = torch.linspace(start=0.0, end=1.0, steps=self.iqn_k + 2, dtype=torch.float, device=device)
                    taus = taus[1:-1].view(1, self.iqn_k)
                    # taus = torch.rand(size=(1, self.iqn_k), dtype=torch.float, device=device)
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
