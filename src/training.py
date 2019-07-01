import sys

sys.path.append("../")

from utils.utils import *
import torch
from PIL import Image
from vizdoom_env.vizdoom_env import DoomEnv
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_episode(env, agent, deterministic, history_length, skip_frames, max_timesteps, normalize_images, state_dim,
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
    state = state_preprocessing(state, height=state_dim[1], width=state_dim[2], normalize=normalize_images)
    image_hist.extend([state] * history_length)
    state = np.array(image_hist).reshape(state_dim)

    beginning = True
    loss, td_loss, L_I, L_F = 0, 0, 0, 0
    trajectory = []
    if type(env) == DoomEnv:
        sector_bbs = create_sector_bounding_box(env.state.sectors)
        visited_sectors = {}

    while True:
        # get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        action_id = agent.act(state=state, deterministic=deterministic)

        # Hint: frame skipping might help you to get better results.
        reward = 0.0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(int(action_id))
            reward += r

            if terminal:
                # Empty multi step buffer to avoid incomplete multi steps in the replay buffer
                agent.n_step_buffer = []
                if type(env) == DoomEnv:
                    return stats, loss, td_loss, L_I, L_F, info, trajectory, env.state.sectors, visited_sectors, sector_bbs
                else:
                    return stats, loss, td_loss, L_I, L_F, info, trajectory, None, None, None

            if rendering:
                env.render()

            if terminal:
                break
        if type(env) == DoomEnv:
            curr_sector = determine_sector(info['x_pos'], info['y_pos'], sector_bbs)
            visited_sectors['section_{}'.format(curr_sector)] = \
                visited_sectors.get('section_{}'.format(curr_sector), 0) + 1
        trajectory = trajectory + [[info['x_pos'], info['y_pos']]]

        next_state = state_preprocessing(next_state, height=state_dim[1], width=state_dim[2],
                                         normalize=normalize_images)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(state_dim)

        if do_training:
            # every transition is added with a high priority such that it gets replayed at least once
            agent.append_to_replay(state=state, action=action_id, next_state=next_state, reward=reward,
                                   terminal=terminal, beginning=beginning, priority=500.0)
            loss, td_loss, L_I, L_F = agent.train()

            # Update the target network
            if not soft_update and agent.steps_done % agent.update_q_target == 0:
                print('Updating Q_target')
                agent.Q_target.load_state_dict(agent.Q.state_dict())

            if step % 100 == 0:
                print('Loss', loss)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:  # or stats.episode_reward < -20:
            if agent.multi_step:
                # Empty multi step buffer to avoid incomplete multi steps in the replay buffer
                agent.n_step_buffer = []
            break
        step += 1
        beginning = False

    print('epsilon_threshold', agent.eps_threshold)

    if type(env) == DoomEnv:
        return stats, loss, td_loss, L_I, L_F, info, trajectory, env.state.sectors, visited_sectors, sector_bbs
    else:
        return stats, loss, td_loss, L_I, L_F, info, trajectory, None, None, None


def train_online(env, agent, writer, num_episodes, eval_cycle, num_eval_episodes, soft_update, skip_frames,
                 history_length, rendering, max_timesteps, normalize_images, state_dim):
    print("... train agent")

    if type(env) == DoomEnv:
        sector_bbs = create_sector_bounding_box(env.state.sectors)

        map_x_min = int(min([sector['x_min'] for _, sector in sector_bbs.items()]))
        map_x_max = int(max([sector['x_max'] for _, sector in sector_bbs.items()]))

        map_y_min = int(min([sector['y_min'] for _, sector in sector_bbs.items()]))
        map_y_max = int(max([sector['y_max'] for _, sector in sector_bbs.items()]))

        map_total_area = sum([sector['area'] for _, sector in sector_bbs.items()])

        uniform_sector_prob = {}
        uniform_dist_sig = np.empty((len(sector_bbs), 3), dtype=np.float32)
        for index, (k, sector) in enumerate(sorted(sector_bbs.items())):
            cord_x = int(sector['cog'][0] - map_x_min)
            cord_y = int(sector['cog'][1] - map_y_min)
            uniform_sector_prob[k] = ((cord_x, cord_y), sector['area'] / map_total_area)
            uniform_dist_sig[index] = np.array([sector['area'] / map_total_area, cord_x, cord_y])
        # uniform_dist_sig = arr_to_sig(uniform_dist)

    for i in range(num_episodes):
        print("episode %d" % i)
        max_timesteps_current = max_timesteps
        stats, loss, td_loss, L_I, L_F, info, trajectory, \
        sectors, visited_sectors, sector_bbs = run_episode(env, agent, max_timesteps=max_timesteps_current,
                                                           deterministic=False,
                                                           do_training=True,
                                                           rendering=rendering,
                                                           soft_update=soft_update,
                                                           skip_frames=skip_frames,
                                                           history_length=history_length,
                                                           normalize_images=normalize_images,
                                                           state_dim=state_dim)

        if len(trajectory) > 0:
            if type(env) == DoomEnv:
                objects = env.state.objects
                writer.add_figure('trajectory', figure=plot_trajectory(trajectory, sectors, sector_bbs, objects),
                                  global_step=i)
                writer.add_scalar('num_visited_sectors', len(visited_sectors), global_step=i)
                writer.add_histogram('visited_sector_ids', [i for i in range(len(env.state.sectors)) if
                                                            'section_{}'.format(i) in visited_sectors.keys()],
                                     global_step=i)

                total_visits = sum([count for _, count in visited_sectors.items()])
                obs_sector_prob = {}
                obs_dist_sig = np.empty((len(sector_bbs), 3), dtype=np.float32)
                for index, (k, sector) in enumerate(sorted(sector_bbs.items())):
                    cord_x = int(sector['cog'][0] - map_x_min)
                    cord_y = int(sector['cog'][1] - map_y_min)
                    obs_sector_prob[k] = ((cord_x, cord_y), visited_sectors.get(k, 0) / total_visits)
                    obs_dist_sig[index] = np.array([visited_sectors.get(k, 0) / total_visits, cord_x, cord_y])
                dist, _, _ = cv2.EMD(obs_dist_sig, uniform_dist_sig, cv2.DIST_L2)
                writer.add_scalar('wasserstein_distance', dist, global_step=i)
            else:
                writer.add_figure('trajectory', figure=plot_trajectory(trajectory, sectors, sector_bbs, None),
                                  global_step=i)

        for key, value in info.items():
            if type(value) is not str:
                writer.add_scalar('info_{}'.format(key), value, global_step=i)

        writer.add_scalar('train_loss', loss, global_step=i)
        writer.add_scalar('train_td_loss', td_loss, global_step=i)
        writer.add_scalar('train_l_i', L_I, global_step=i)
        writer.add_scalar('train_l_f', L_F, global_step=i)
        writer.add_scalar('train_episode_reward', stats.episode_reward, global_step=i)
        for action in range(env.action_space.n):
            writer.add_scalar('train_{}'.format(action), stats.get_action_usage(action), global_step=i)

        # EVALUATION
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        if i % eval_cycle == 0:
            stats = []
            for j in range(num_eval_episodes):
                stats.append(run_episode(env, agent, deterministic=True, do_training=False, max_timesteps=1000,
                                         history_length=history_length, skip_frames=skip_frames,
                                         normalize_images=normalize_images, state_dim=state_dim)[0])
            stats_agg = [stat.episode_reward for stat in stats]
            episode_reward_mean, episode_reward_std = np.mean(stats_agg), np.std(stats_agg)
            print('Validation {} +- {}'.format(episode_reward_mean, episode_reward_std))
            print('Replay buffer length', agent.replay_buffer.size)
            writer.add_scalar('val_episode_reward_mean', episode_reward_mean, global_step=i)
            writer.add_scalar('val_episode_reward_std', episode_reward_std, global_step=i)

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            model_dir = agent.save(os.path.join(writer.logdir, "agent.pt"))
            print("Model saved in file: %s" % model_dir)


def state_preprocessing(state, height, width, normalize=True):
    image_resized = Image.fromarray(state).resize((height, width), Image.ANTIALIAS)
    image_resized_bw = rgb2gray(np.array(image_resized))
    if normalize:
        image_resized_bw = image_resized_bw / 255.0
    return image_resized_bw
