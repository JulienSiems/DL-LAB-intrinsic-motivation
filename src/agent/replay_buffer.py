import numpy as np


class UniformReplayBuffer:
    def __init__(self, capacity, state_shape, state_store_dtype, state_sample_dtype, *args, **kwargs):
        """
        Replay buffer with uniform sampling.
        Assumes consecutively added transitions.
        :param capacity: The capacity of the replay buffer.
        :param state_dim: The shape of a single state. Like (B, C, ...) with B = 1.
        :param state_store_dtype: The dtype of a state when stored in the buffer.
        :param state_sample_dtype: The dtype of a state when sampled from the buffer.
        """
        assert capacity >= 1, "Capacity must be >= 1"
        assert len(state_shape) >= 3, "Expected state shape like (B, C, ...)"
        self.capacity = capacity
        self.state_shape = state_shape
        self.state_store_dtype = state_store_dtype
        self.state_sample_dtype = state_sample_dtype
        buffer_state_shape = list(self.state_shape)
        buffer_state_shape[0] = self.capacity
        self.states = np.zeros(shape=buffer_state_shape, dtype=self.state_store_dtype)
        self.actions = np.zeros(shape=(self.capacity,), dtype=np.long)
        self.rewards = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.states_next = np.zeros(shape=buffer_state_shape, dtype=self.state_store_dtype)
        self.dones = np.zeros(shape=capacity, dtype=np.bool)
        self.episode_idxs = np.full(shape=capacity, fill_value=-1, dtype=np.long)
        self.size = 0
        self.cur_idx = 0
        self.cur_episode_idx = -1

    def propagate_up(self, idx, priority):
        pass

    def add_transition(self, s, a, r, s_next, done, beginning, *args, **kwargs):
        """
        Add new transition to replay buffer.
        :param s: State
        :param a: Action
        :param r: Reward
        :param s_next: Next state
        :param done: Done flag
        :param beginning: Flag which indicates a new episode just begun. If true, increment cur_episode_idx.
        """
        if beginning:
            self.cur_episode_idx += 1
        self.states[self.cur_idx] = s
        self.actions[self.cur_idx] = a
        self.rewards[self.cur_idx] = r
        self.states_next[self.cur_idx] = s_next
        self.dones[self.cur_idx] = done
        self.episode_idxs[self.cur_idx] = self.cur_episode_idx
        self.cur_idx = (self.cur_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size, history_length, n_steps, *args, **kwargs):
        """
        Sample a batch of transitions.
        :param batch_size: Batch size
        :param history_length: Number of stacked frames per state.
        :param n_steps: Number of successive states. Useful for multi-step return.
        :return:
        - batch_state_sequences: The sampled states, with shape (batch_size, n_steps, history_length * n_channels, ...)
        - batch_actions: The sampled actions, with shape (batch_size, n_steps)
        - batch_rewards: The sampled rewards, with shape (batch_size, n_steps)
        - batch_dones: The sampled done flags, with shape (batch_size, n_steps)
        - actual_n_steps: Not every sample can contain the full n_steps, e.g. if episode terminates before n_steps
            are reached. The actual_n_steps can be used to calculate the proper multi-step reward.
        - weights: Not important for uniform sampling...
        - sample_idxs: Not important for uniform sampling...
        """
        # assert batch_size >= 1
        # assert history_length >= 1
        # assert n_steps >= 1
        sample_idxs = np.random.choice(self.size, batch_size)
        batch_state_sequences_shape = [batch_size, n_steps + 1, self.state_shape[1] * history_length]
        batch_state_sequences_shape += list(self.state_shape)[2:]
        batch_state_sequences = np.zeros(shape=batch_state_sequences_shape, dtype=self.state_sample_dtype)
        batch_actions = np.zeros(shape=(batch_size, n_steps), dtype=np.long)
        batch_rewards = np.zeros(shape=(batch_size, n_steps), dtype=np.float32)
        batch_dones = np.zeros(shape=(batch_size, n_steps), dtype=np.bool)
        actual_n_steps = np.zeros(shape=(batch_size,), dtype=np.int)
        weights = np.ones(shape=(batch_size,), dtype=np.float32)
        for batch_idx in range(batch_size):
            sample_idx = sample_idxs[batch_idx]
            sample_episode_idx = self.episode_idxs[sample_idx]
            # construct state history for first state. following states can reuse part of previous state history.
            offset_sample_idx = sample_idx
            for history_step in range(history_length - 1, -1, -1):
                h_start = history_step * self.state_shape[1]
                h_end = h_start + self.state_shape[1]
                batch_state_sequences[batch_idx, 0, h_start:h_end, ...] = self.states[offset_sample_idx]
                # go back one history step, if predecessor belongs to same episode.
                new_offset_sample_idx = (offset_sample_idx - 1) % self.capacity
                if self.episode_idxs[new_offset_sample_idx] == sample_episode_idx:
                    offset_sample_idx = new_offset_sample_idx
            # copy reusable part of history from previous state history to next state history and add new frame at end.
            prev_h_start = self.state_shape[1]
            cur_h_end = (history_length - 1) * self.state_shape[1]
            offset_sample_idx = sample_idx
            for step in range(n_steps):
                prev_state_part = batch_state_sequences[batch_idx, step, prev_h_start:, ...]
                batch_state_sequences[batch_idx, step + 1, :cur_h_end, ...] = prev_state_part
                batch_state_sequences[batch_idx, step + 1, cur_h_end:, ...] = self.states_next[offset_sample_idx]
                batch_actions[batch_idx, step] = self.actions[offset_sample_idx]
                batch_rewards[batch_idx, step] = self.rewards[offset_sample_idx]
                batch_dones[batch_idx, step] = self.dones[offset_sample_idx]
                # if next state does not belong to episode, break and track how many steps actually were done.
                new_offset_sample_idx = (offset_sample_idx + 1) % self.capacity
                if self.episode_idxs[new_offset_sample_idx] != sample_episode_idx:
                    actual_n_steps[batch_idx] = step + 1
                    break
                else:
                    offset_sample_idx = new_offset_sample_idx
            else:
                # step loop did not break -> all n_steps were done.
                actual_n_steps[batch_idx] = n_steps
        return batch_state_sequences, batch_actions, batch_rewards, batch_dones, actual_n_steps, weights, sample_idxs


class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, state_store_dtype, state_sample_dtype, alpha, *args, **kwargs):
        """
        Replay buffer with proportional sampling.
        Assumes consecutively added transitions.
        Need capacity to be a power of 2 due to sum tree implementation.
        :param capacity: The capacity of the replay buffer.
        :param state_dim: The shape of a single state. Like (B, C, ...) with B = 1.
        :param state_store_dtype: The dtype of a state when stored in the buffer.
        :param state_sample_dtype: The dtype of a state when sampled from the buffer.
        :param alpha: Interpolate between uniform (0) and fully proportional (1) sampling.
        """
        assert capacity >= 1 and capacity & (capacity - 1) == 0, "Capacity must be >= 1 and a power of 2."
        assert len(state_shape) >= 3, "Expected state shape like (B, C, ...)"
        assert 0 <= alpha <= 1, "Alpha must be in range [0, 1]"
        self.capacity = capacity
        self.state_shape = state_shape
        self.state_store_dtype = state_store_dtype
        self.state_sample_dtype = state_sample_dtype
        self.alpha = alpha
        buffer_state_shape = list(self.state_shape)
        buffer_state_shape[0] = self.capacity
        self.states = np.zeros(shape=buffer_state_shape, dtype=self.state_store_dtype)
        self.actions = np.zeros(shape=(self.capacity,), dtype=np.long)
        self.rewards = np.zeros(shape=(self.capacity,), dtype=np.float32)
        self.states_next = np.zeros(shape=buffer_state_shape, dtype=self.state_store_dtype)
        self.dones = np.zeros(shape=capacity, dtype=np.bool)
        self.episode_idxs = np.full(shape=capacity, fill_value=-1, dtype=np.long)
        self.size = 0
        self.cur_idx = 0
        self.cur_episode_idx = -1
        self.sum_tree = np.zeros(shape=(self.capacity * 2,), dtype=np.double)
        self.sum_tree_depth = int(np.log2(self.capacity * 2))
        self.priority_eps = 1e-2  # added to every transition priority -> non-zero probability for all

    def propagate_up(self, idx, priority):
        """
        When inserting or changing a transition, propagate the change bottom-up through sum tree.
        :param idx: Index of transition.
        :param priority: New priority of transition.
        """
        idx += self.capacity  # Transform index from buffer view to sum tree view.
        diff = priority - self.sum_tree[idx]
        self.sum_tree[idx] = priority
        for _ in range(self.sum_tree_depth - 1):
            idx //= 2
            self.sum_tree[idx] += diff

    def add_transition(self, s, a, r, s_next, done, beginning, priority, *args, **kwargs):
        """
        Add new transition to replay buffer.
        :param s: State
        :param a: Action
        :param r: Reward
        :param s_next: Next state
        :param done: Done flag
        :param beginning: Flag which indicates a new episode just begun. If true, increment cur_episode_idx.
        :param priority: Priority of transition.
        """
        # assert priority >= 0
        if beginning:
            self.cur_episode_idx += 1
        self.states[self.cur_idx] = s
        self.actions[self.cur_idx] = a
        self.rewards[self.cur_idx] = r
        self.states_next[self.cur_idx] = s_next
        self.dones[self.cur_idx] = done
        self.episode_idxs[self.cur_idx] = self.cur_episode_idx
        priority = (priority + self.priority_eps) ** self.alpha
        self.propagate_up(self.cur_idx, priority)
        self.cur_idx = (self.cur_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_idx(self, p_left, p_right):
        """
        Sample a transition idx from sum tree propotional to its priority.
        Only sample from range [p_left, p_right].
        :param p_left: Left border of a subinterval of range [0, 1].
        :param p_right: Right border of a subinterval of range [0, 1].
        :return: transition index in state buffer and its priority.
        """
        val = np.random.uniform(p_left, p_right) * self.sum_tree[1]
        pos = 1
        l_border = 0.0
        for _ in range(self.sum_tree_depth - 1):
            l_pos = pos * 2
            l_val = l_border + self.sum_tree[l_pos]
            if val <= l_val:
                pos = l_pos
            else:
                pos = l_pos + 1
                l_border = l_val
        return (pos - self.capacity), self.sum_tree[pos]

    def sample_batch(self, batch_size, history_length, n_steps, beta, balance_batch, *args, **kwargs):
        """
        Sample a batch of transitions.
        :param batch_size: Batch size
        :param history_length: Number of stacked frames per state.
        :param n_steps: Number of successive states. Useful for multi-step return.
        :param beta: Interpolate between full compensation for non-uniform sampling (1) and no compensation (0).
        :param balance_batch: If true, split range [0, 1] into batch_size equally long intervals and sample with them.
        :return:
        - batch_state_sequences: The sampled states, with shape (batch_size, n_steps, history_length * n_channels, ...)
        - batch_actions: The sampled actions, with shape (batch_size, n_steps)
        - batch_rewards: The sampled rewards, with shape (batch_size, n_steps)
        - batch_dones: The sampled done flags, with shape (batch_size, n_steps)
        - actual_n_steps: Not every sample can contain the full n_steps, e.g. if episode terminates before n_steps
            are reached. The actual_n_steps can be used to calculate the proper multi-step reward.
        - weights: Weights which are used to weigh the errors of the samples when updating model.
        - sample_idxs: The sample indices which are used to update the priorities in the sum tree, after having
            calculated new priorities when updating the model.
        """
        # assert batch_size >= 1
        # assert history_length >= 1
        # assert n_steps >= 1
        if balance_batch:
            ls = np.linspace(0.0, 0.98, batch_size + 1)
            sample_idxs_and_priorities = [self._sample_idx(ls[bidx], ls[bidx+1]) for bidx in range(batch_size)]
        else:
            sample_idxs_and_priorities = [self._sample_idx(0.0, 0.98) for _ in range(batch_size)]
        sample_idxs = np.array([tmp[0] for tmp in sample_idxs_and_priorities])
        priorities = np.array([tmp[1] for tmp in sample_idxs_and_priorities], dtype=np.float32)
        weights = np.power(((priorities + self.priority_eps) / self.sum_tree[1]) * self.size, -beta)
        weights = weights / (np.max(weights) + 1e-8)  # normalize weights to be max 1.0
        weights = np.array(weights, dtype=np.float32)
        batch_state_sequences_shape = [batch_size, n_steps + 1, self.state_shape[1] * history_length]
        batch_state_sequences_shape += list(self.state_shape)[2:]
        batch_state_sequences = np.zeros(shape=batch_state_sequences_shape, dtype=self.state_sample_dtype)
        batch_actions = np.zeros(shape=(batch_size, n_steps), dtype=np.long)
        batch_rewards = np.zeros(shape=(batch_size, n_steps), dtype=np.float32)
        batch_dones = np.zeros(shape=(batch_size, n_steps), dtype=np.bool)
        actual_n_steps = np.zeros(shape=(batch_size,), dtype=np.int)
        for batch_idx in range(batch_size):
            sample_idx = sample_idxs[batch_idx]
            sample_episode_idx = self.episode_idxs[sample_idx]
            # construct state history for first state. following states can reuse part of previous state history.
            offset_sample_idx = sample_idx
            for history_step in range(history_length - 1, -1, -1):
                h_start = history_step * self.state_shape[1]
                h_end = h_start + self.state_shape[1]
                batch_state_sequences[batch_idx, 0, h_start:h_end, ...] = self.states[offset_sample_idx]
                # go back one history step, if predecessor belongs to same episode.
                new_offset_sample_idx = (offset_sample_idx - 1) % self.capacity
                if self.episode_idxs[new_offset_sample_idx] == sample_episode_idx:
                    offset_sample_idx = new_offset_sample_idx
            # copy reusable part of history from previous state history to next state history and add new frame at end.
            prev_h_start = self.state_shape[1]
            cur_h_end = (history_length - 1) * self.state_shape[1]
            offset_sample_idx = sample_idx
            for step in range(n_steps):
                prev_state_part = batch_state_sequences[batch_idx, step, prev_h_start:, ...]
                batch_state_sequences[batch_idx, step + 1, :cur_h_end, ...] = prev_state_part
                batch_state_sequences[batch_idx, step + 1, cur_h_end:, ...] = self.states_next[offset_sample_idx]
                batch_actions[batch_idx, step] = self.actions[offset_sample_idx]
                batch_rewards[batch_idx, step] = self.rewards[offset_sample_idx]
                batch_dones[batch_idx, step] = self.dones[offset_sample_idx]
                # if next state does not belong to episode, break and track how many steps actually were done.
                new_offset_sample_idx = (offset_sample_idx + 1) % self.capacity
                if self.episode_idxs[new_offset_sample_idx] != sample_episode_idx:
                    actual_n_steps[batch_idx] = step + 1
                    break
                else:
                    offset_sample_idx = new_offset_sample_idx
            else:
                # step loop did not break -> all n_steps were done.
                actual_n_steps[batch_idx] = n_steps
        return batch_state_sequences, batch_actions, batch_rewards, batch_dones, actual_n_steps, weights, sample_idxs
