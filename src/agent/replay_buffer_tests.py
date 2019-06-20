from replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer
import numpy as np
import time

# test uniform
state_shape = (1, 3, 2, 2)
rb = UniformReplayBuffer(8, state_shape, np.float16, np.float32)

# first episode
rb.add_transition(np.full(shape=state_shape, fill_value=0, dtype=np.float32), 0, 0.0,
                  np.full(shape=state_shape, fill_value=1, dtype=np.float32), False, True, 0.0)
rb.add_transition(np.full(shape=state_shape, fill_value=1, dtype=np.float32), 1, 1.0,
                  np.full(shape=state_shape, fill_value=2, dtype=np.float32), False, False, 0.0)
rb.add_transition(np.full(shape=state_shape, fill_value=2, dtype=np.float32), 2, 2.0,
                  np.full(shape=state_shape, fill_value=3, dtype=np.float32), True, False, 0.0)

# second episode
rb.add_transition(np.full(shape=state_shape, fill_value=10, dtype=np.float32), 10, 10.0,
                  np.full(shape=state_shape, fill_value=11, dtype=np.float32), False, True, 0.0)
rb.add_transition(np.full(shape=state_shape, fill_value=11, dtype=np.float32), 11, 11.0,
                  np.full(shape=state_shape, fill_value=12, dtype=np.float32), False, False, 0.0)
rb.add_transition(np.full(shape=state_shape, fill_value=12, dtype=np.float32), 12, 12.0,
                  np.full(shape=state_shape, fill_value=13, dtype=np.float32), False, False, 0.0)
rb.add_transition(np.full(shape=state_shape, fill_value=13, dtype=np.float32), 13, 13.0,
                  np.full(shape=state_shape, fill_value=14, dtype=np.float32), True, False, 0.0)
# sample and print
states, action, reward, done, actual_n, weights, sidxs = rb.sample_batch(batch_size=2, history_length=2, n_steps=3)
print(states)
print(states.shape)
print(action)
print(reward)
print(done)
print(actual_n)
print(weights)
print(sidxs)
print()

# third episode
rb.add_transition(np.full(shape=state_shape, fill_value=20, dtype=np.float32), 20, 20.0,
                  np.full(shape=state_shape, fill_value=21, dtype=np.float32), False, True, 0.0)
rb.add_transition(np.full(shape=state_shape, fill_value=21, dtype=np.float32), 21, 21.0,
                  np.full(shape=state_shape, fill_value=22, dtype=np.float32), False, False, 0.0)
rb.add_transition(np.full(shape=state_shape, fill_value=22, dtype=np.float32), 22, 22.0,
                  np.full(shape=state_shape, fill_value=23, dtype=np.float32), True, False, 0.0)

# sample and print
states, action, reward, done, actual_n, weights, sidxs = rb.sample_batch(batch_size=2, history_length=2, n_steps=3)
print(states)
print(states.shape)
print(action)
print(reward)
print(done)
print(actual_n)
print(weights)
print(sidxs)
print()

# test proportional
state_shape = (1, 3, 2, 2)
rb = PrioritizedReplayBuffer(8, state_shape, np.float16, np.float32, alpha=1.0)

# first episode
rb.add_transition(np.full(shape=state_shape, fill_value=0, dtype=np.float32), 0, 0.0,
                  np.full(shape=state_shape, fill_value=1, dtype=np.float32), False, True, 40.0)
rb.add_transition(np.full(shape=state_shape, fill_value=1, dtype=np.float32), 1, 1.0,
                  np.full(shape=state_shape, fill_value=2, dtype=np.float32), False, False, 20.0)
rb.add_transition(np.full(shape=state_shape, fill_value=2, dtype=np.float32), 2, 2.0,
                  np.full(shape=state_shape, fill_value=3, dtype=np.float32), False, False, 10.0)
rb.add_transition(np.full(shape=state_shape, fill_value=3, dtype=np.float32), 3, 3.0,
                  np.full(shape=state_shape, fill_value=4, dtype=np.float32), True, False, 10.0)

# second episode
rb.add_transition(np.full(shape=state_shape, fill_value=10, dtype=np.float32), 10, 10.0,
                  np.full(shape=state_shape, fill_value=11, dtype=np.float32), False, True, 7.0)
rb.add_transition(np.full(shape=state_shape, fill_value=11, dtype=np.float32), 11, 11.0,
                  np.full(shape=state_shape, fill_value=12, dtype=np.float32), False, False, 7.0)
rb.add_transition(np.full(shape=state_shape, fill_value=12, dtype=np.float32), 12, 12.0,
                  np.full(shape=state_shape, fill_value=13, dtype=np.float32), True, False, 6.0)

# sample and print
states, action, reward, done, actual_n, weights, sidxs = rb.sample_batch(batch_size=4, history_length=1,
                                                                         n_steps=3, beta=1.0, balance_batch=True)
print(states)
print(states.shape)
print(action)
print(reward)
print(done)
print(actual_n)
print(weights)
print(sidxs)
print()

# check distribution
batch_size = 2048
states, action, reward, done, actual_n, weights, sidxs = rb.sample_batch(batch_size=batch_size, history_length=1,
                                                                         n_steps=1, beta=1.0, balance_batch=True)
bins = [0 for _ in range(7)]
for idx in sidxs:
    bins[idx] += 1

print([tmp/batch_size for tmp in bins])
print()

# compare batch variance with and without batch_balance
n_batches = 100
batch_size = 16
bins_with = np.zeros(shape=(n_batches, 7), dtype=np.int)
bins_without = np.zeros(shape=(n_batches, 7), dtype=np.int)
for b_idx in range(n_batches):
    states, action, reward, done, actual_n, weights, sidxs = rb.sample_batch(batch_size=batch_size,
                                                                             history_length=1,
                                                                             n_steps=1, beta=1.0,
                                                                             balance_batch=True)
    for idx in sidxs:
        bins_with[b_idx, idx] += 1
    states, action, reward, done, actual_n, weights, sidxs = rb.sample_batch(batch_size=batch_size,
                                                                             history_length=1,
                                                                             n_steps=1, beta=1.0,
                                                                             balance_batch=False)
    for idx in sidxs:
        bins_without[b_idx, idx] += 1

print(np.var(bins_with, axis=0))
print(np.var(bins_without, axis=0))
print()

# check distribution after changing priorities
rb.propagate_up(0, 20.0)
rb.propagate_up(6, 26.0)
batch_size = 2048
states, action, reward, done, actual_n, weights, sidxs = rb.sample_batch(batch_size=batch_size, history_length=1,
                                                                         n_steps=1, beta=1.0, balance_batch=True)
bins = [0 for _ in range(7)]
for idx in sidxs:
    bins[idx] += 1

print([tmp / batch_size for tmp in bins])
print()

# stress test
state_shape = (1, 1, 42, 42)
sample_every = 4
batch_size = 32
history_length = 4
n_steps = 3
rb = PrioritizedReplayBuffer(2**19, state_shape, np.float16, np.float32, 0.5)
dummy_state = np.zeros(shape=state_shape, dtype=np.float32)
done = False
beginning = True
t0 = time.time()
for i in range(5 * 2**19):
    done = np.random.uniform() > 0.95
    rb.add_transition(dummy_state, 0, 0.0, dummy_state, done, beginning, 0.5)
    beginning = True if done else False
    if i % sample_every == 0:
        rb.sample_batch(batch_size, history_length, n_steps, 0.5, balance_batch=True)
    if i % 2**17 == 0:
        print(i, time.time() - t0)
