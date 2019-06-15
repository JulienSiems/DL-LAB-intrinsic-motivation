import tensorflow as tf
import torch
import numpy as np
from agent.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4, state_dim=0, act_dist=None, do_training=False, replay_buffer_size=10000):
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

        self.epsilon_decay_end = 5 * 10 ** 3
        self.epsilon_start = 0.9
        self.epsilon_final = 0.05
        self.epsilon_max = 1.0

        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
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

        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)
        batch_states = torch.from_numpy(batch_states).float().to(self.device)
        batch_actions = torch.from_numpy(batch_actions).long().to(self.device)
        batch_next_states = torch.from_numpy(batch_next_states).float().to(self.device)
        batch_rewards = torch.from_numpy(batch_rewards).float().to(self.device)
        batch_dones = torch.from_numpy(batch_dones.astype(int)).byte().to(self.device)

        self.set_train_mode()
        self.optimizer.zero_grad()
        _, target_actions = self.Q(batch_next_states).max(dim=1)
        q_target_values = self.Q_target(batch_next_states)
        target_action_values = q_target_values.gather(dim=1, index=target_actions.unsqueeze(-1)).squeeze(-1)
        target_action_values[batch_dones] = 0.0
        td_target = batch_rewards + self.gamma * target_action_values
        q_values = self.Q(batch_states)
        action_values = q_values.gather(dim=1, index=batch_actions.unsqueeze(-1)).squeeze(-1)

        loss = self.loss_function(action_values, td_target.detach())
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
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # take greedy action (argmax)
            state = torch.from_numpy(state).float().to(self.device)
            _, action_id = self.Q(state).view(1, -1).max(dim=1)
            action_id = action_id.item()
        else:

            # sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.choice(self.num_actions, p=self.action_probs)
          
        return action_id

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


