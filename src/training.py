import sys

sys.path.append("../")

from utils.utils import *
import torch
from PIL import Image
from vizdoom_env.vizdoom_env import DoomEnv
import cv2
import glob
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_episode(env, agent, deterministic, history_length, skip_frames, max_timesteps, normalize_images, state_dim,
                init_prio, do_training, rendering, soft_update):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """
    stats = EpisodeStats()

    step = 0
    state = env.reset()
    done = False
    action = 0
    beginning = True

    # fix bug of corrupted states without rendering in gym environment
    # env.viewer.window.dispatch_events()

    # create history buffer and fill it with first state
    history_buffer = np.zeros(shape=(state_dim[0] * history_length, state_dim[1], state_dim[2]), dtype=np.float32)
    state = state_preprocessing(state, height=state_dim[1], width=state_dim[2], normalize=normalize_images)
    for h_idx in range(0, state_dim[0] * history_length, state_dim[0]):
        history_buffer[h_idx] = state

    losses = []  # each entry like (loss, td_loss, L_I, L_F)
    action_values = []
    trajectory = []
    if type(env) == DoomEnv:
        sector_bbs = create_sector_bounding_box(env.state.sectors)
        visited_sectors = {}

    while not done and step < max_timesteps:
        # get action from agent every (skip_frames + 1) frames when training or every frame when not training
        if step % (skip_frames + 1) == 0 or not do_training:
            act_res = agent.act(state=history_buffer, deterministic=deterministic)
            action = int(act_res[0])
            action_values.append(act_res[1].flatten())

        next_state, reward, done, info = env.step(action)

        if rendering:
            env.render()

        if type(env) == DoomEnv:
            curr_sector = determine_sector(info['x_pos'], info['y_pos'], sector_bbs)
            visited_sectors['section_{}'.format(curr_sector)] = \
                visited_sectors.get('section_{}'.format(curr_sector), 0) + 1

        if agent.intrinsic:
            # if there is intrinsic reward to track, keep old state history to calculate intrinsic reward for this step
            history_buffer_old = history_buffer.copy()[np.newaxis, ...]

        next_state = state_preprocessing(next_state, height=state_dim[1], width=state_dim[2],
                                         normalize=normalize_images)
        # update history buffer with latest state. shift all states one to the left and add latest state at end.
        for h_idx in range(state_dim[0], state_dim[0] * history_length, state_dim[0]):
            history_buffer[(h_idx - state_dim[0]):h_idx] = history_buffer[h_idx:(h_idx + state_dim[0])]
        history_buffer[state_dim[0] * (history_length - 1):] = next_state

        if agent.intrinsic:
            with torch.no_grad():
                _, _, r_i, _ = agent.intrinsic_reward_generator.compute_intrinsic_reward(history_buffer_old,
                                                                                         np.array([action]),
                                                                                         history_buffer[
                                                                                             np.newaxis, ...])

        if 'x_pos' in info and 'y_pos' in info:
            if agent.intrinsic:
                trajectory = trajectory + [[info['x_pos'], info['y_pos'], r_i.item()]]
            else:
                trajectory = trajectory + [[info['x_pos'], info['y_pos'], 0]]

        if do_training:
            # when initially added, a transition gets assigned a high priority, such that it gets replayed at least once
            agent.replay_buffer.add_transition(state, action, reward, next_state, done, beginning, init_prio)

            train_res = agent.train()
            if train_res[0] is not None:
                losses.append(train_res)

            # update the target network
            agent.steps_done += 1
            if not soft_update and agent.steps_done % agent.update_q_target == 0:
                print('updating Q_target')
                agent.Q_target.load_state_dict(agent.Q.state_dict())

            if (step + 1) % 100 == 0:
                print('step: {}\tloss: {:.4f}\ttd_loss: {:.4f}\tL_I: {:.4f}\tL_F: {:.4f}'.format(step + 1, *losses[-1]))

        stats.step(reward, r_i.item() * agent.mu if agent.intrinsic else 0.0, action)
        state = next_state
        step += 1
        beginning = False

    if type(env) == DoomEnv:
        return stats, losses, info, trajectory, action_values, env.state.sectors, visited_sectors, sector_bbs
    else:
        return stats, losses, info, trajectory, action_values, None, None, None


def train_online(env, agent, writer, num_episodes, eval_cycle, num_eval_episodes, soft_update, skip_frames,
                 history_length, rendering, max_timesteps, normalize_images, state_dim, init_prio, num_model_files,
                 simple_coverage_threshold, geometric_coverage_gamma, num_total_steps, store_cycle):
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
        cumulative_obs_sector_visits = {}
        cumulative_obs_sector_total_visits = 0
        # Initialize the coverage metric
        coverage_metrics = Coverage(num_sectors=len(uniform_sector_prob))

    # practically infinite episodes if num_episodes is not >= 1. also add 1 to num_episodes because agent is not trained
    # for last episode. this way, agent is evaluated num_episodes + 1 times but trained num_episodes times
    # (only matters if num_episodes is the determining criterion for number of runs)
    num_episodes = num_episodes + 1 if num_episodes >= 1 else sys.maxsize

    # quite similar to num_total_steps
    num_total_steps = num_total_steps if num_total_steps >= 1 else sys.maxsize

    total_steps = 0
    for episode_idx in range(num_episodes):
        is_last_episode = episode_idx >= (num_episodes - 1) or total_steps >= num_total_steps

        # EVALUATION
        # check its performance with greedy actions only
        if eval_cycle >= 1 and num_eval_episodes >= 1 and (episode_idx % eval_cycle == 0 or is_last_episode):
            stats = []
            for j in range(num_eval_episodes):
                stats.append(run_episode(env, agent, deterministic=True, do_training=False, max_timesteps=max_timesteps,
                                         history_length=history_length, skip_frames=skip_frames,
                                         normalize_images=normalize_images, state_dim=state_dim,
                                         init_prio=init_prio, rendering=rendering, soft_update=False)[0])
            episode_rewards = [stat.episode_reward for stat in stats]
            episode_reward_mean, episode_reward_std = np.mean(episode_rewards), np.std(episode_rewards)
            print('Validation {} +- {}'.format(episode_reward_mean, episode_reward_std))
            print('Replay buffer length', agent.replay_buffer.size)
            writer.add_scalar('val_episode_reward_mean', episode_reward_mean, global_step=episode_idx)
            writer.add_scalar('val_episode_reward_std', episode_reward_std, global_step=episode_idx)

        # store model.
        if store_cycle >= 1 and (episode_idx % store_cycle == 0 or is_last_episode):
            if num_model_files >= 1:
                model_files = glob.glob(os.path.join(writer.logdir, "agent*"))
                # Sort by date
                model_files.sort(key=os.path.getmtime)
                if len(model_files) > num_model_files - 1:
                    # Delete the oldest model file
                    os.remove(model_files[0])
            agent.save(os.path.join(writer.logdir, "agent_{}.pt".format(episode_idx)))
            print('model saved!')

        if is_last_episode:
            break

        # training episode
        print("episode %d" % episode_idx)
        max_timesteps_current = max_timesteps
        stats, losses, info, trajectory, action_values, \
        sectors, visited_sectors, sector_bbs = run_episode(env, agent, max_timesteps=max_timesteps_current,
                                                           deterministic=False,
                                                           do_training=True,
                                                           rendering=rendering,
                                                           soft_update=soft_update,
                                                           skip_frames=skip_frames,
                                                           history_length=history_length,
                                                           normalize_images=normalize_images,
                                                           state_dim=state_dim,
                                                           init_prio=init_prio)

        if len(trajectory) > 0:
            if type(env) == DoomEnv:
                objects = env.state.objects
                writer.add_figure('trajectory', figure=plot_trajectory(trajectory, sectors, sector_bbs, objects,
                                                                       intrinsic=agent.intrinsic),
                                  global_step=episode_idx)
                writer.add_scalar('num_visited_sectors', len(visited_sectors), global_step=episode_idx)

                simple_coverage, geometric_coverage, occupancy_density_entropy = \
                    coverage_metrics.compute_coverage(visited_sectors=visited_sectors,
                                                      K=simple_coverage_threshold * skip_frames,
                                                      gamma=geometric_coverage_gamma)
                writer.add_scalar('simple_coverage', simple_coverage, global_step=episode_idx)
                writer.add_scalar('geometric_coverage', geometric_coverage, global_step=episode_idx)
                writer.add_scalar('occupancy_density_entropy', occupancy_density_entropy, global_step=episode_idx)

                writer.add_histogram('visited_sector_ids', [i for i in range(len(env.state.sectors)) if
                                                            'section_{}'.format(i) in visited_sectors.keys()],
                                     global_step=episode_idx)

                total_visits = sum([count for _, count in visited_sectors.items()])
                obs_sector_prob = {}
                obs_dist_sig = np.empty((len(sector_bbs), 3), dtype=np.float32)
                cumulative_obs_dist_sig = np.empty((len(sector_bbs), 3), dtype=np.float32)
                for index, (k, sector) in enumerate(sorted(sector_bbs.items())):
                    cord_x = int(sector['cog'][0] - map_x_min)
                    cord_y = int(sector['cog'][1] - map_y_min)
                    obs_sector_prob[k] = ((cord_x, cord_y), visited_sectors.get(k, 0) / total_visits)
                    obs_dist_sig[index] = np.array([visited_sectors.get(k, 0) / total_visits, cord_x, cord_y])
                    if cumulative_obs_sector_total_visits > 0:
                        cumulative_obs_dist_sig[index] = np.array(
                            [(cumulative_obs_sector_visits.get(k, (0, 0))[1] / cumulative_obs_sector_total_visits),
                             cord_x, cord_y])

                if cumulative_obs_sector_total_visits > 0:
                    dist, _, _ = cv2.EMD(obs_dist_sig, cumulative_obs_dist_sig, cv2.DIST_L2)
                    writer.add_scalar('wasserstein_distance (current trajectory vs past trajectories)', dist,
                                      global_step=episode_idx)

                for index, (k, sector) in enumerate(sorted(sector_bbs.items())):
                    cord_x = int(sector['cog'][0] - map_x_min)
                    cord_y = int(sector['cog'][1] - map_y_min)
                    cumulative_obs_sector_visits[k] = (
                        (cord_x, cord_y), cumulative_obs_sector_visits.get(k, (0, 0))[1] + visited_sectors.get(k, 0))
                cumulative_obs_sector_total_visits += total_visits

                current_cumulative_obs_dist_sig = np.empty((len(sector_bbs), 3), dtype=np.float32)
                for index, (k, sector) in enumerate(sorted(sector_bbs.items())):
                    cord_x = int(sector['cog'][0] - map_x_min)
                    cord_y = int(sector['cog'][1] - map_y_min)
                    current_cumulative_obs_dist_sig[index] = np.array(
                        [cumulative_obs_sector_visits.get(k, (0, 0))[1] / cumulative_obs_sector_total_visits, cord_x,
                         cord_y])

                dist, _, _ = cv2.EMD(obs_dist_sig, uniform_dist_sig, cv2.DIST_L2)
                writer.add_scalar('wasserstein_distance (current trajectory vs uniform disttribution)', dist,
                                  global_step=episode_idx)

                dist, _, _ = cv2.EMD(current_cumulative_obs_dist_sig, uniform_dist_sig, cv2.DIST_L2)
                writer.add_scalar('wasserstein_distance (cumulative trajectory vs uniform disttribution)', dist,
                                  global_step=episode_idx)
            else:
                tmp_figure = plot_trajectory(trajectory, sectors, sector_bbs, None, intrinsic=agent.intrinsic)
                writer.add_figure('trajectory', figure=tmp_figure, global_step=episode_idx)

            # Append current trajectory to save of all trajectories
            with open(os.path.join(writer.logdir, "trajectories.obj"), 'ab+') as fp:
                pickle.dump(trajectory, fp)

        for key, value in info.items():
            if type(value) is not str:
                writer.add_scalar('info_{}'.format(key), value, global_step=episode_idx)

        writer.add_scalar('train_loss', np.mean([losses[i][0] for i in range(len(losses))]), global_step=episode_idx)
        writer.add_scalar('train_td_loss', np.mean([losses[i][1] for i in range(len(losses))]), global_step=episode_idx)
        writer.add_scalar('train_l_i', np.mean([losses[i][2] for i in range(len(losses))]), global_step=episode_idx)
        writer.add_scalar('train_l_f', np.mean([losses[i][3] for i in range(len(losses))]), global_step=episode_idx)
        writer.add_scalar('train_episode_reward', stats.episode_reward, global_step=episode_idx)
        writer.add_scalar('train_episode_length', stats.steps, global_step=episode_idx)
        writer.add_scalar('intrinsic_episode_reward', stats.intrinsic_reward, global_step=episode_idx)
        for action in range(env.action_space.n):
            writer.add_scalar('action_freq_{}'.format(action), stats.get_action_usage(action), global_step=episode_idx)
            mean_action_value = np.mean([action_values[i][action] for i in range(len(action_values))])
            writer.add_scalar('action_val_{}'.format(action), mean_action_value, global_step=episode_idx)

        total_steps += stats.steps


def state_preprocessing(state, height, width, normalize=True):
    # watch out, PIL's resize uses (width, height) format!
    # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.resize
    image_resized = Image.fromarray(state).resize((width, height), Image.ANTIALIAS)
    image_resized_bw = rgb2gray(np.array(image_resized))
    if normalize:
        image_resized_bw = image_resized_bw / 255.0
    return image_resized_bw
