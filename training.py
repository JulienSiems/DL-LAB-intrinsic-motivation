import sys

sys.path.append("../")

from utils.utils import *
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_episode(env, agent, deterministic, history_length, skip_frames, max_timesteps, normalize_images,
                do_training=True, rendering=False, soft_update=False):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    # env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state, normalize=normalize_images)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape([history_length + 1, 42, 42])

    possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

    while True:
        # get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state=state, deterministic=deterministic)
        action = possible_actions[action_id]

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape([history_length + 1, 42, 42])

        if do_training:
            agent.append_to_replay(state=state, action=action_id, next_state=next_state, reward=reward,
                                   terminal=terminal)
            agent.train()

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps: #  or stats.episode_reward < -20:
            if agent.multi_step:
                # Finish n step buffer
                agent.finish_n_step()
            break
        step += 1

    print('epsilon_threshold', agent.eps_threshold)

    # Update the target network
    if not soft_update:
        agent.Q_target.load_state_dict(agent.Q.state_dict())

    return stats


def train_online(env, agent, writer, num_episodes, eval_cycle, num_eval_episodes, soft_update, skip_frames,
                 history_length, rendering, max_timesteps, normalize_images):
    print("... train agent")

    for i in range(num_episodes):
        print("episode %d" % i)
        max_timesteps_current = max_timesteps
        stats = run_episode(env, agent, max_timesteps=max_timesteps_current, deterministic=False, do_training=True,
                            rendering=rendering, soft_update=soft_update, skip_frames=skip_frames,
                            history_length=history_length, normalize_images=normalize_images)

        writer.add_scalar('train_episode_reward', stats.episode_reward, global_step=i)
        writer.add_scalar('train_straight', stats.get_action_usage(STRAIGHT), global_step=i)
        writer.add_scalar('train_left', stats.get_action_usage(LEFT), global_step=i)
        writer.add_scalar('train_right', stats.get_action_usage(RIGHT), global_step=i)
        writer.add_scalar('train_accel', stats.get_action_usage(ACCELERATE), global_step=i)
        writer.add_scalar('train_brake', stats.get_action_usage(BRAKE), global_step=i)

        # EVALUATION
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            stats = []
            for j in range(num_eval_episodes):
                stats.append(run_episode(env, agent, deterministic=True, do_training=False, max_timesteps=1000,
                                         history_length=history_length, skip_frames=skip_frames,
                                         normalize_images=normalize_images))
            stats_agg = [stat.episode_reward for stat in stats]
            episode_reward_mean, episode_reward_std = np.mean(stats_agg), np.std(stats_agg)
            print('Validation {} +- {}'.format(episode_reward_mean, episode_reward_std))
            print('Replay buffer length', len(agent.replay_buffer._data))
            writer.add_scalar('val_episode_reward_mean', episode_reward_mean, global_step=i)
            writer.add_scalar('val_episode_reward_std', episode_reward_std, global_step=i)

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            model_dir = agent.save(os.path.join(writer.logdir, "agent.pt"))
            print("Model saved in file: %s" % model_dir)


def state_preprocessing(state, normalize=True):
    image_resized = Image.fromarray(state).resize((42, 42), Image.ANTIALIAS)
    image_resized_bw = rgb2gray(np.array(image_resized))
    if normalize:
        image_resized_bw = image_resized_bw / 255.0
    return image_resized_bw
