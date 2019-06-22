from collections import namedtuple
import numpy as np
import os
import gzip
import pickle

# class BufferData(np.object):
#     def __init__(self):
#         self.state = 0
#         self.action = 0
#         self.next_state = 0
#         self.reward = 0
#         self.done = 0
#
#     def set_data(self, state, action, next_state, reward, done):
#         self.state = state
#         self.action = action
#         self.next_state = next_state
#         self.reward = reward
#         self.done = done


class ReplayBuffer:

    # implement a capacity for the replay buffer (FIFO, capacity: 1e5 - 1e6)

    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, state_dim, buffer_size):
        self.buffer_size = int(buffer_size)
        self.dt = np.dtype([('state', np.float32, state_dim), ('action', np.float32, (1,)),
                            ('next_state', np.float32, state_dim), ('reward', np.float32, (1,)), ('done', np.bool, (1,))])
        self.data = np.zeros((self.buffer_size,), dtype=self.dt)
        self.data_counter = 0
        self.data_read_counter = 0

    def add_transition(self, state, action, next_state, reward, done):
        """
        This method adds a transition to the replay buffer.
        """
        self.data[self.data_counter]['state'] = state
        self.data[self.data_counter]['action'] = action
        self.data[self.data_counter]['next_state'] = next_state
        self.data[self.data_counter]['reward'] = reward
        self.data[self.data_counter]['done'] = done

        self.data_counter += 1
        if self.data_counter >= self.buffer_size:
            self.data_read_counter = self.buffer_size
            self.data_counter = self.data_counter % self.buffer_size
        elif self.data_read_counter != self.buffer_size:
            self.data_read_counter = self.data_counter

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        batch_indices = np.random.choice(self.data_read_counter, batch_size)
        # batch_indices = random.sample(self.data_read_counter, k=batch_size)

        batch_states = np.array([self.data[i]['state'] for i in batch_indices])
        batch_actions = np.array([self.data[i]['action'] for i in batch_indices]).squeeze()
        batch_next_states = np.array([self.data[i]['next_state'] for i in batch_indices])
        batch_rewards = np.array([self.data[i]['reward'] for i in batch_indices]).squeeze()
        batch_dones = np.array([self.data[i]['done'] for i in batch_indices]).squeeze()
        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones


import random
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBufferMaster:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, capacity):
        self._data = deque(maxlen=30000)

    def add_transition(self, *args):
        """
        This method adds a transition to the replay buffer.
        """
        self._data.append(Transition(*args))

    def next_batch(self, batch_size):
        """
        This method samples a batch of transitions.
        """
        # Samples batch
        batch = random.sample(self._data, k=batch_size)  # List of Tuples
        # Used in https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html,
        # adapted from https://stackoverflow.com/a/19343
        batch_transitions = Transition(*zip(*batch))  # Tuple of Lists

        return [np.array(item) for item in batch_transitions]
