import random

import numpy as np
import torch

from src.agent.replay_buffer import ReplayBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, intrinsic_reward_generator, num_actions, capacity, intrinsic, extrinsic,
                 mu, beta, lambda_intrinsic, multi_step=True, multi_step_size=3, non_uniform_sampling=False, gamma=0.95,
                 batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4, number_replays=10, loss_function='L1',
                 soft_update=False, algorithm='DQN', epsilon_schedule=False, pre_intrinsic=True, **kwargs):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tao: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q
        self.Q_target = Q_target
        self.Q_target.load_state_dict(self.Q.state_dict())

        # intrinsic reward generator
        self.intrinsic_reward_generator = intrinsic_reward_generator

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=capacity)

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
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.num_actions = num_actions
        self.steps_done = 0
        self.soft_update = soft_update
        self.non_uniform_sampling = non_uniform_sampling

        self.algorithm = algorithm

        self.epsilon_schedule = epsilon_schedule
        self.n_step_buffer = []
        self.n_steps = multi_step_size
        self.multi_step = multi_step

        if loss_function == 'L1':
            self.loss_function = torch.nn.SmoothL1Loss()
        elif loss_function == 'L2':
            self.loss_function = torch.nn.MSELoss()
        else:
            raise ValueError('Loss function {} not implemented.'.format(loss_function))

    # Adapated from https://github.com/qfettes/DeepRL-Tutorials/blob/master/02.NStep_DQN.ipynb
    def append_to_replay(self, state, action, next_state, reward, terminal):
        if self.multi_step:
            self.n_step_buffer.append((state, action, next_state, reward))

            if len(self.n_step_buffer) < self.n_steps:
                return

            R = sum([self.n_step_buffer[i][3] * (self.gamma ** i) for i in range(self.n_steps)])
            state, action, _, _ = self.n_step_buffer.pop(0)

            self.replay_buffer.add_transition(state, action, next_state, R, terminal)
        else:
            self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

    def finish_n_step(self):
        if self.multi_step:
            while len(self.n_step_buffer) > 0:
                R = sum([self.n_step_buffer[i][3] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
                state, action, next_state, _ = self.n_step_buffer.pop(0)

                self.replay_buffer.add_transition(state, action, next_state, R, True)

    def train(self):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        if self.batch_size > len(self.replay_buffer._data):
            return None, None, None, None

        for iter in range(self.number_replays):
            # 2. sample next batch and perform batch update: (initially take less than batch_size because of the replay
            # buffer
            batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = \
                self.replay_buffer.next_batch(batch_size=self.batch_size)

            # Set the expected value of a terminated section to zero.
            non_final_mask = torch.from_numpy(np.array(batch_dones != True, dtype=np.uint8))
            non_final_next_states = torch.from_numpy(batch_next_states)[non_final_mask].to(device).float()

            #       2.1 compute td targets and loss
            #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
            next_state_values = torch.zeros(self.batch_size, device=device)
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
                L_I, L_F, intrinsic_reward = self.intrinsic_reward_generator.compute_intrinsic_reward(
                    state=batch_states,
                    action=batch_actions,
                    next_state=batch_next_states)
                intrinsic_reward = intrinsic_reward.detach() * self.mu
            else:
                L_I, L_F, intrinsic_reward = [torch.tensor([0], dtype=torch.float, device=device) for _ in range(3)]

            if self.extrinsic or self.pre_intrinsic:
                extrinsic_reward = torch.from_numpy(batch_rewards).to(device).float()
            else:
                extrinsic_reward = 0

            # print(intrinsic_reward[0])
            reward = extrinsic_reward + (intrinsic_reward if not self.pre_intrinsic else 0)

            # Detach from comp graph to avoid that gradients are propagated through the target network.
            next_state_values = next_state_values.detach()
            if self.multi_step:
                td_target = reward + (self.gamma ** self.n_steps) * next_state_values
            else:
                td_target = reward + self.gamma * next_state_values

            #       2.2 update the Q network
            #              self.Q.update(...)
            state_action_values = self.Q(torch.from_numpy(batch_states).to(device).float())
            batch_actions_tensor = torch.from_numpy(batch_actions).to(device).view(-1, 1)

            # Choose the action previously taken
            q_pick = torch.gather(state_action_values, dim=1, index=batch_actions_tensor.long())
            # Chosen like in this tutorial https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            self.optimizer.zero_grad()
            # print(L_I.item(), L_F.item())
            # td_loss = self.loss_function(input=q_pick, target=td_target.unsqueeze(1))
            td_loss = (q_pick-td_target.detach()).pow(2).mean()
            loss = self.lambda_intrinsic * td_loss + (1 - self.beta) * L_I + self.beta * L_F
            loss.backward()
            self.optimizer.step()

        #       2.3 call soft update for target network
        if self.soft_update:
            soft_update(self.Q_target, self.Q, self.tau)

        return loss.item(), td_loss.item(), L_I.item(), L_F.item()

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
            EPS_START = 0.9
            EPS_END = 0.05
            EPS_DECAY = 30000
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            np.exp(-1. * self.steps_done / EPS_DECAY)
            self.steps_done += 1
        else:
            eps_threshold = self.epsilon
        self.eps_threshold = eps_threshold
        r = np.random.uniform()
        if deterministic or sample > eps_threshold:
            with torch.no_grad():
                # take greedy action (argmax)
                action_soft = self.Q(torch.from_numpy(np.expand_dims(state, 0)).to(device).float())
                action_id = torch.argmax(action_soft).detach().cpu().numpy()
        else:
            if self.non_uniform_sampling:
                action_id = \
                    np.random.choice(self.num_actions, 1, p=[0.45, 0.15, 0.15, 0.15, 0.1])[0]
            else:
                action_id = np.random.randint(self.num_actions)
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name, map_location='cpu'))
        self.Q_target.load_state_dict(torch.load(file_name, map_location='cpu'))
