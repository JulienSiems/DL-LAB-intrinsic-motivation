import random
from collections import deque, namedtuple

import numpy as np

# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self, capacity):
        self._data = deque(maxlen=capacity)

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
