import random

import numpy as np
import torch

from src.agent.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, intrinsic_reward_generator, num_actions, capacity, lr, intrinsic, extrinsic,
                 mu, beta, lambda_intrinsic, epsilon_start, epsilon_end, epsilon_decay, update_q_target, history_length,
                 experience_replay, prio_er_alpha, prio_er_beta_start, prio_er_beta_end, prio_er_beta_decay, init_prio,
                 state_dim, iqn, iqn_n, iqn_np, iqn_k, iqn_det_max_train, iqn_det_max_act, huber_kappa, epsilon, tau,
                 n_step_reward, train_every_n_steps, train_n_times, non_uniform_sampling, gamma, batch_size,
                 soft_update, ddqn, epsilon_schedule, pre_intrinsic, nu_action_probs, adam_epsilon, *args, **kwargs):
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
        self.optimizer = torch.optim.Adam(parameters, lr=lr, eps=adam_epsilon)

        # define replay buffer
        tmp_state_shape = tuple([1, state_dim[0], state_dim[1], state_dim[2]])
        self.experience_replay = experience_replay
        self.prio_er_alpha = prio_er_alpha
        self.prio_er_beta_start = prio_er_beta_start
        self.prio_er_beta_end = prio_er_beta_end
        self.prio_er_beta_decay = prio_er_beta_decay
        self.cur_prio_er_beta = prio_er_beta_start
        self.init_prio = init_prio
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
        self.batch_size_range = np.arange(self.batch_size)
        self.gamma = gamma
        self.tau = tau
        self.mu = mu
        self.beta = beta
        self.lambda_intrinsic = lambda_intrinsic
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.pre_intrinsic = pre_intrinsic

        self.history_length = history_length
        self.train_every_n_steps = train_every_n_steps
        self.train_n_times = train_n_times
        self.num_actions = num_actions
        self.steps_done = 0
        self.train_steps = 0
        self.train_steps_done = 0
        self.soft_update = soft_update
        self.non_uniform_sampling = non_uniform_sampling
        self.nu_action_probs = nu_action_probs

        self.ddqn = ddqn

        self.cur_epsilon = epsilon_start if epsilon_schedule else epsilon
        self.epsilon_schedule = epsilon_schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.update_q_target = update_q_target
        self.n_step_reward = n_step_reward
        self.reward_gamma_mask = np.array([self.gamma ** t for t in range(self.n_step_reward)], dtype=np.float32)

        self.iqn = iqn
        self.iqn_n = iqn_n
        self.iqn_np = iqn_np
        self.iqn_k = iqn_k
        self.iqn_det_max_train = iqn_det_max_train
        self.iqn_det_max_act = iqn_det_max_act
        self.huber_kappa = huber_kappa

    def train(self):
        # advance prioritized experience replay beta
        interpolate_b = min(1.0, self.train_steps / self.prio_er_beta_decay)
        self.cur_prio_er_beta = self.prio_er_beta_start * (1.0 - interpolate_b) + self.prio_er_beta_end * interpolate_b

        # advance exploration epsilon
        if self.epsilon_schedule:
            interpolate_eps = min(1.0, self.train_steps / self.epsilon_decay)
            self.cur_epsilon = self.epsilon_start * (1.0 - interpolate_eps) + self.epsilon_end * interpolate_eps

        if self.train_steps % self.train_every_n_steps != 0 or self.batch_size > self.replay_buffer.size:
            self.train_steps += 1
            return None, None, None, None

        for train_iter_idx in range(self.train_n_times):
            self.optimizer.zero_grad()

            batch_state_sequences, batch_actions, batch_rewards, batch_dones, actual_n_steps, weights, sample_idxs = \
                self.replay_buffer.sample_batch(self.batch_size, history_length=self.history_length, balance_batch=True,
                                                n_steps=self.n_step_reward, beta=self.cur_prio_er_beta)
            batch_states = batch_state_sequences[:, 0, ...]
            batch_imm_next_states = batch_state_sequences[:, 1, ...]  # immediate next states for ICM module
            batch_next_states = batch_state_sequences[(self.batch_size_range, actual_n_steps)]
            batch_actions = batch_actions[:, 0]
            batch_rewards = np.sum(batch_rewards * self.reward_gamma_mask, axis=1)
            batch_dones = batch_dones[(self.batch_size_range, actual_n_steps - 1)]
            weights = torch.from_numpy(weights).detach().to(device)

            batch_states_ = torch.from_numpy(batch_states).to(device)
            batch_next_states_ = torch.from_numpy(batch_next_states).to(device)
            batch_actions_ = torch.from_numpy(batch_actions).long().to(device).view(-1, 1)
            batch_rewards_ = torch.from_numpy(batch_rewards).to(device).float()
            actual_n_steps_ = torch.from_numpy(actual_n_steps).to(device).float()

            # compute mask that weighs values of terminated next states with zero.
            non_final_mask = torch.from_numpy(np.array(batch_dones != True, dtype=np.float32)).to(device)

            if self.iqn:
                # when using IQN, net outputs are of shape (batch_size * num_taus, num_actions)
                taus = torch.rand(size=(self.batch_size, self.iqn_n), dtype=torch.float, device=device)
                taus_prime = torch.rand(size=(self.batch_size, self.iqn_np), dtype=torch.float, device=device)
                if self.iqn_det_max_train:
                    taus_tilde = torch.linspace(0.0, 1.0, steps=self.iqn_k + 2, dtype=torch.float, device=device)
                    taus_tilde = taus_tilde[1:-1].view(1, self.iqn_k).repeat(self.batch_size, 1)
                else:
                    taus_tilde = torch.rand(size=(self.batch_size, self.iqn_k), dtype=torch.float, device=device)
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
                    next_state=batch_imm_next_states)
                # this is not detached because it is used to optimise the forward model.
                r_i = intrinsic_reward
                # this is detached because it's used for the reward to optimise the Q model.
                intrinsic_reward = intrinsic_reward.detach() * self.mu
            else:
                L_I, L_F, intrinsic_reward, l_i = \
                    [torch.tensor([0], dtype=torch.float, device=device) for _ in range(4)]

            if self.extrinsic or self.pre_intrinsic:
                extrinsic_reward = batch_rewards_
            else:
                extrinsic_reward = 0.0

            reward = extrinsic_reward + (intrinsic_reward if not self.pre_intrinsic else 0.0)
            if self.iqn:
                reward = reward.view(-1, 1).repeat(1, self.iqn_np).view(-1, 1).squeeze()
                actual_n_steps_ = actual_n_steps_.view(-1, 1).repeat(1, self.iqn_np).view(-1, 1).squeeze()

            td_target = reward + (self.gamma ** actual_n_steps_) * Q_target_pred_next_picked

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

        self.train_steps += 1
        self.train_steps_done += 1
        return losses.mean().item(), td_losses.mean().item(), L_I.item(), L_F.item()

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to
        select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        with torch.no_grad():
            state_ = torch.from_numpy(np.expand_dims(state, 0)).to(device).float()
            if self.iqn:
                # for IQN we have to sample from reward distribution to determine action values
                if self.iqn_det_max_act:
                    taus = torch.linspace(0.0, 1.0, steps=self.iqn_k + 2, dtype=torch.float, device=device)
                    taus = taus[1:-1].view(1, self.iqn_k)
                else:
                    taus = torch.rand(size=(1, self.iqn_k), dtype=torch.float, device=device)
                pred = self.Q(state_, taus=taus).mean(dim=0)
            else:
                pred = self.Q(state_)
            if deterministic or random.random() > self.cur_epsilon:
                # take greedy action (argmax)
                action_id = torch.argmax(pred).detach().cpu().numpy()
            else:
                if self.non_uniform_sampling:
                    action_id = np.random.choice(self.num_actions, size=(1,), p=self.nu_action_probs)[0]
                else:
                    action_id = np.random.randint(self.num_actions, size=(1,))
        return action_id, pred.detach().cpu().numpy()

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name, map_location='cpu'))
        self.Q_target.load_state_dict(torch.load(file_name, map_location='cpu'))
