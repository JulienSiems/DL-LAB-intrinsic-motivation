import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
from agent.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from agent.networks import *

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4, state_dim=0,
                 act_dist=None, do_training=False, replay_buffer_size=10000, icm_beta=0.2, icm_lambda=0.1, icm_eta=20,
                 use_icm=True, use_extrinsic_reward=True, policy='e_greedy'):
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
        if torch.cuda.is_available():
            device_name = 'cuda'
        else:
            device_name = 'cpu'
        self.device = torch.device(device_name)

        self.use_icm = use_icm
        self.use_extrinsic_reward = use_extrinsic_reward
        self.policy = policy.lower()

        self.Q = Q.to(self.device)
        self.Q_target = Q_target.to(self.device)

        self.Q_target.init_zero()
        # self.Q_target.load_state_dict(self.Q.state_dict())

        if do_training:
            # define replay buffer
            self.replay_buffer = ReplayBuffer(state_dim=state_dim, buffer_size=replay_buffer_size)
        
        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.icm_beta = icm_beta
        self.icm_lambda = icm_lambda
        self.icm_eta = icm_eta

        self.epsilon_decay_end = 5 * 10 ** 3
        self.epsilon_start = 0.9
        self.epsilon_final = 0.05
        self.epsilon_max = 1.0

        self.epsilon = epsilon

        self.q_loss_fun = torch.nn.MSELoss()
        self.forward_loss_fun = torch.nn.MSELoss()
        # self.inverse_loss_fun = torch.nn.CrossEntropyLoss()
        self.inverse_loss_fun = torch.nn.NLLLoss()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

        if act_dist is not None:
            self.action_probs = act_dist
        else:
            self.action_probs = [1 / self.num_actions] * self.num_actions

        self.Q_target.eval()

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #              self.Q.update(...)
        #       2.3 call soft update for target network
        #              self.Q_target.update(...)

        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        batch_states, batch_action_ids, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        batch_states = torch.from_numpy(batch_states).float().to(self.device)
        batch_action_ids = torch.from_numpy(batch_action_ids).long().to(self.device)
        batch_next_states = torch.from_numpy(batch_next_states).float().to(self.device)
        batch_rewards = torch.from_numpy(batch_rewards).float().to(self.device)
        batch_dones = torch.from_numpy(batch_dones.astype(int)).byte().to(self.device)

        batch_actions = torch.zeros((batch_action_ids.shape[0], self.num_actions))
        batch_actions[:, batch_action_ids] = 1.0

        self.set_train_mode()
        self.optimizer.zero_grad()

        inv_out, forward_out, q_out = self.Q(state=batch_states, next_state=batch_next_states, action=batch_actions, mode=ICM_GET_ALL_OUT)
        action_values = q_out.gather(dim=1, index=batch_action_ids.unsqueeze(-1)).squeeze(-1)

        _, _, q_out = self.Q(state=batch_next_states, next_state=None, action=None, mode=ICM_GET_ONLY_Q_OUT)
        _, target_actions = q_out.max(dim=1)

        _, _, qtarget_out = self.Q_target(state=batch_next_states, next_state=None, action=None, mode=ICM_GET_ONLY_Q_OUT)
        q_target_values = qtarget_out

        target_action_values = q_target_values.gather(dim=1, index=target_actions.unsqueeze(-1)).squeeze(-1)
        target_action_values[batch_dones] = 0.0

        batch_next_states = self.Q.encode(batch_next_states)

        intrinsic_reward = (self.icm_eta * (forward_out - batch_next_states).pow(2).sum()) / 2
        batch_rewards += intrinsic_reward.detach()

        td_target = batch_rewards + self.gamma * target_action_values

        policy_loss = (action_values-td_target.detach()).pow(2).mean()

        if self.use_icm:
            # inverse_loss = self.inverse_loss_fun(inv_out, batch_action_ids)
            inverse_loss = self.inverse_loss_fun(F.softmax(inv_out, dim=1), batch_action_ids)
            forward_loss = 0.5 * (forward_out-batch_next_states).pow(2).mean() * batch_next_states.shape[1]

            loss = (self.icm_lambda) * policy_loss + (1 - self.icm_beta) * inverse_loss + self.icm_beta * forward_loss
            # print(policy_loss.item(), inverse_loss.item(), forward_loss.item(), loss.item())
        else:
            loss = policy_loss
            # print(policy_loss.item(), 0, 0, loss.item())

        loss.backward()
        self.optimizer.step()

        soft_update(self.Q_target, self.Q, self.tau)

        self.set_eval_mode()

        return loss

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        if deterministic:
            r = 1
        elif self.policy == 'random':
            r = 0
        else:
            r = np.random.uniform()

        if deterministic or r > self.epsilon:
            # take greedy action (argmax)
            state = torch.from_numpy(state).float().to(self.device)
            inv_out, forward_out, q_out = self.Q(state=state, next_state=None, action=None, mode=ICM_GET_ONLY_Q_OUT)
            _, action_id = q_out.view(1, -1).max(dim=1)
            action_id = action_id.item()
        else:
            # sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.choice(self.num_actions, p=self.action_probs)
          
        return action_id

    def get_intrinsic_reward(self, state, next_state, action):
        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        inv_out, forward_out, q_out = self.Q(state=state, next_state=next_state, action=action, mode=ICM_GET_ONLY_FORWARD_OUT)
        next_state = self.Q.encode(next_state)
        reward = (self.icm_eta * (forward_out-next_state).pow(2).sum()) / 2
        return reward.item()

    def save(self, data, file_name):
        torch.save(data, file_name)

    def load(self, file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint_data = torch.load(file_name, map_location=self.device)
        self.Q.load_state_dict(checkpoint_data['Qnet_final'])
        self.Q_target.load_state_dict(checkpoint_data['Qtargetnet_final'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_final'])
        return checkpoint_data

    def load_best_configuration(self, file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint_data = torch.load(file_name, map_location=self.device)
        self.Q.load_state_dict(checkpoint_data['Qnet'])
        self.Q_target.load_state_dict(checkpoint_data['Qtargetnet'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer'])
        return checkpoint_data

    def set_train_mode(self):
        self.Q.train()

    def set_eval_mode(self):
        self.Q.eval()


