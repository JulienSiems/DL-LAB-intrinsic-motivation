import inspect
import json
import socket
from datetime import datetime

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tensorboardX import SummaryWriter
import math
import itertools
import cv2

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4

sns.set_style('whitegrid')


def compute_accuracy(y_pred, y_gt):
    return (y_pred == y_gt).sum().item() / y_pred.shape[0]


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.expand_dims(np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721]), axis=0)
    return gray.astype('float32')


def setup_experiment_folder_writer(frame, name, log_dir=None, args_for_filename=None, additional_param=None):
    """
    Create experiment folder and tensorboardx writer
    :param args_for_filename:
    :param name:
    :param log_dir:
    :param frame: Inspection frame of the main function.
    :return: tensorboardx writer
    """
    args, _, _, values = inspect.getargvalues(frame)
    if additional_param is not None:
        args.extend([k for k in additional_param.keys()])
    if args_for_filename is None:
        args_for_filename = args
    comment = name + ''.join(['_{}_{}_'.format(arg, values[arg]) for arg in args if arg in args_for_filename])
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # Create tensorboardx writer
    if log_dir:
        writer = SummaryWriter(logdir=os.path.join(log_dir, '_'.join([current_time, socket.gethostname(), comment])))
    else:
        writer = SummaryWriter(comment=comment)
    # Dump config of current run to the experiment folder
    config_dict = {arg: values[arg] for arg in args}
    with open(os.path.join(writer.logdir, 'config.json'), 'w') as fp:
        json.dump(config_dict, fp)
    # Add config as text summary
    writer.add_text('config', str(config_dict), global_step=0)
    return writer


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]):
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.intrinsic_reward = 0.0
        self.actions_ids = []
        self.steps = 0

    def step(self, reward, intrinsic_reward, action_id):
        self.episode_reward += reward
        self.intrinsic_reward += intrinsic_reward
        self.actions_ids.append(action_id)
        self.steps += 1

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)


def plot_trajectory(trajectory, sectors, sector_bb, objects, intrinsic):
    trajectory = np.array(trajectory)

    fig = plt.figure()

    if sectors is not None:
        for i, s in enumerate(sectors):
            # Plot sector on map
            for l in s.lines:
                if l.is_blocking:
                    plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)

        # Add section number to figure.
        for sector_id, (_, coor) in enumerate(sector_bb.items()):
            plt.text((coor['x_min'] + coor['x_max']) / 2, (coor['y_min'] + coor['y_max']) / 2, str(sector_id),
                     horizontalalignment='center', verticalalignment='center')

    if objects is not None:
        for o in objects:
            # Plot object on map
            if o.name == "DoomPlayer":
                plt.plot(o.position_x, o.position_y, color='green', marker='o', label='Player')
            else:
                plt.plot(o.position_x, o.position_y, color='red', marker='x', label='Goal')

    if intrinsic:
        intrinsic_rewards = np.array([x.item() for x in trajectory[:, 2]])
        intrinsic_rewards = intrinsic_rewards / intrinsic_rewards.max()
        trajectory_plot = plt.scatter(trajectory[:, 0], trajectory[:, 1],
                                      color=[plt.cm.magma(intrinsic_rewards[i]) for i in range(len(trajectory))],
                                      label='Exploration', marker='o', linestyle='dashed')
        trajectory_plot.set_array(np.linspace(0, 1, len(trajectory)))
        cbar = fig.colorbar(trajectory_plot, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.set_yticklabels([str(i) for i in np.arange(start=0, stop=len(trajectory) + 1, step=20)])
        cbar.ax.set_ylabel('Curiosity', rotation=270)
    else:
        trajectory_plot = plt.scatter(trajectory[:, 0], trajectory[:, 1],
                                      color=[plt.cm.magma(i) for i in np.linspace(0, 1, len(trajectory))],
                                      label='Exploration', marker='o', linestyle='dashed')
        trajectory_plot.set_array(np.linspace(0, 1, len(trajectory)))
        cbar = fig.colorbar(trajectory_plot, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.set_yticklabels([str(i) for i in np.arange(start=0, stop=len(trajectory) + 1, step=20)])
        cbar.ax.set_ylabel('Action step', rotation=270)



    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.legend()

    plt.axis('equal')
    plt.tight_layout()
    return fig


def determine_sector(x, y, sector_bbs):
    for i, (section_key, section_value) in enumerate(sector_bbs.items()):
        # Determine which section the agent is in
        if (section_value['x_min'] <= x <= section_value['x_max']) and (
                section_value['y_min'] <= y <= section_value['y_max']):
            return i

    raise ValueError('The sector the agent is in could not be found.')


def create_sector_bounding_box(sectors):
    sector_bbs = {}
    for i, sector_i in enumerate(sectors):
        x_min = min(min([[line.x1, line.x2] for line in sector_i.lines]))
        x_max = max(max([[line.x1, line.x2] for line in sector_i.lines]))

        y_min = min(min([[line.y1, line.y2] for line in sector_i.lines]))
        y_max = max(max([[line.y1, line.y2] for line in sector_i.lines]))

        # vertices = []
        # last_vertices = [[0, 0], [0, 0]]
        # for j, line in enumerate(sector_i.lines):
        #     if j == 0:
        #         vertices.append((line.x1, line.y1))
        #         vertices.append((line.x2, line.y2))
        #         last_vertices = [[line.x1, line.y1], [line.x2, line.y2]]
        #     else:
        #         if (last_vertices[0][0] != line.x1 or last_vertices[0][1] != line.y1) and \
        #                 (last_vertices[1][0] != line.x1 or last_vertices[1][1] != line.y1):
        #             vertices.append((line.x1, line.y1))
        #         if (last_vertices[0][0] != line.x2 or last_vertices[0][1] != line.y2) and \
        #                 (last_vertices[1][0] != line.x2 or last_vertices[1][1] != line.y2):
        #             vertices.append((line.x2, line.y2))
        #         last_vertices = [[line.x1, line.y1], [line.x2, line.y2]]

        sector_bbs['section_{}'.format(i)] = {
            'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'area': (x_max - x_min) * (y_max - y_min),
            'cog': ((x_max + x_min) / 2, (y_max + y_min) / 2)
        }

    # TODO: Calculate area of non rectangular rooms. Currently fixed for vizdoom.
    sector_bbs['section_16']['area'] = 15360
    sector_bbs['section_16']['cog'] = (488, -640)

    return sector_bbs


# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolygonArea(corners):
    n = len(corners)  # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


class Coverage:
    def __init__(self, num_sectors: int):
        self.num_sectors = num_sectors
        self.visited_sectors_list = [0 for _ in range(num_sectors)]
        self.max_entropy = (1 / self.num_sectors) * math.log2(1 / self.num_sectors) * self.num_sectors

    def compute_coverage(self, visited_sectors: dict, K: int, gamma: float) -> int:
        """Computes the coverage metrics outlined on slack."""
        # Update the current list of visited sectors
        self.visited_sectors_list = [
            self.visited_sectors_list[i] + visited_sectors.get('section_{}'.format(i), 0) for i in
            range(self.num_sectors)]

        # Compute simple coverage
        covered_sectors = list(map(lambda x: int(x >= K), self.visited_sectors_list))

        # Compute geometric coverage
        geometric_weighted_sectors = list(map(lambda x: 1 - gamma ** x, self.visited_sectors_list))

        # Compute Occupation Density Entropy
        total_visits = sum(self.visited_sectors_list)
        occupancy_density_entropy = list(
            map(lambda x: (x / total_visits) * math.log2(x / total_visits) if x != 0 else 0, self.visited_sectors_list))

        return sum(covered_sectors) / self.num_sectors, sum(geometric_weighted_sectors) / self.num_sectors, sum(
            occupancy_density_entropy) / self.max_entropy

class ExplorationMetrics:
    def __init__(self, num_sectors: int, alpha: float, sector_bbs: dict):
        self.alpha = alpha
        self.num_sectors = num_sectors
        self.visited_sectors_list = [0 for _ in range(num_sectors)]
        self.max_entropy = (1 / self.num_sectors) * math.log2(1 / self.num_sectors) * self.num_sectors
        self.current_eval_visited_sectors_list = [0 for _ in range(num_sectors)]
        self.cumulative_visited_sectors_list = [0 for _ in range(num_sectors)]
        self.total_visits = 0
        self.eval_count = 0
        self.current_eval_visit_prob_list = []
        self.sector_bbs = sector_bbs
        self.map_x_min = int(min([sector['x_min'] for _, sector in sector_bbs.items()]))
        self.map_x_max = int(max([sector['x_max'] for _, sector in sector_bbs.items()]))
        self.map_y_min = int(min([sector['y_min'] for _, sector in sector_bbs.items()]))
        self.map_y_max = int(max([sector['y_max'] for _, sector in sector_bbs.items()]))

    def add_evaluation(self, visited_sectors: dict):
        self.current_eval_visited_sectors_list = [
            self.current_eval_visited_sectors_list[i] + visited_sectors.get('section_{}'.format(i), 0) for i in
            range(self.num_sectors)]
        self.eval_count += 1
        visit_count = sum([visited_sectors.get('section_{}'.format(i), 0) for i in range(self.num_sectors)])
        current_eval_visit_prob = [visited_sectors.get('section_{}'.format(i), 0) / visit_count for i in range(self.num_sectors)]
        self.current_eval_visit_prob_list.append(current_eval_visit_prob)

    def get_entropy(self, prob):
        return list(map(lambda x: -1 * (x / total_visits) * math.log2(x / total_visits) if x != 0 else 0, prob))

    def get_cross_entropy(self, x, y, max_entropy):
        if x == 0 and y == 0:
            return 0
        elif y == 0:
            return max_entropy
        else:
            return -1 * x * math.log2(y)

    def get_total_variance_distance(self, p, q):
        total_variance = sum(list(map(lambda x, y: abs(x - y), p, q))) / 2
        return total_variance

    def get_wasserstein_distance(self, p, q):
        p_sig = np.empty((self.num_sectors, 3), dtype=np.float32)
        q_sig = np.empty((self.num_sectors, 3), dtype=np.float32)
        for i in range(self.num_sectors):
            cord_x = int(self.sector_bbs['section_{}'.format(i)]['cog'][0] - self.map_x_min)
            cord_y = int(self.sector_bbs['section_{}'.format(i)]['cog'][1] - self.map_y_min)
            p_sig[i] = np.array([p[i], cord_x, cord_y])
            q_sig[i] = np.array([q[i], cord_x, cord_y])
        dist, _, _ = cv2.EMD(p_sig, q_sig, cv2.DIST_L2)
        return dist

    def compute_metrics(self) -> int:
        """Computes the coverage metrics outlined on slack."""
        expectation_current_policy_p = list(map(lambda x: x / self.eval_count, self.current_eval_visited_sectors_list))

        eval_visits = sum(self.current_eval_visited_sectors_list) / self.eval_count
        normalized_current_policy_p = list(map(lambda x: x / eval_visits, expectation_current_policy_p))

        prev_visits = sum(self.cumulative_visited_sectors_list)
        if prev_visits > 0:
            normalized_cumulative_previous_policies_p = list(map(lambda x: x / prev_visits, self.cumulative_visited_sectors_list))

            # Entropy score and gain
            p = list(map(lambda x, y: self.alpha * x + (1 - self.alpha) * y,
                         normalized_current_policy_p, normalized_cumulative_previous_policies_p))

            policy_score = -1 * sum(list(map(lambda x: x * math.log2(x) if x != 0 else 0, p)))
            cumulative_policy_score = -1 * sum(list(map(lambda x: x * math.log2(x) if x != 0 else 0, normalized_cumulative_previous_policies_p)))
            policy_gain = policy_score - cumulative_policy_score

            # Cross Entropy
            a = list(map(lambda x: -1 * math.log2(x) if x != 0 else 0, normalized_cumulative_previous_policies_p))
            max_cross_entropy = max(list(map(lambda x: -1 * math.log2(x) if x != 0 else 0, normalized_cumulative_previous_policies_p)))

            cross_entropy = sum(list(map(
                lambda x, y, m: self.get_cross_entropy(x, y, m), normalized_current_policy_p,
                normalized_cumulative_previous_policies_p,
                itertools.repeat(max_cross_entropy, len(normalized_cumulative_previous_policies_p))
            )))
        else:
            policy_score = 0
            policy_gain = 0
            cross_entropy = 0

        # Exploration variance
        total_variance = sum(
            list(map(
                lambda x, y: self.get_total_variance_distance(x, y), self.current_eval_visit_prob_list,
                     itertools.repeat(normalized_current_policy_p, len(self.current_eval_visit_prob_list))
            ))
        ) / (self.eval_count - 1)

        # wasserstein variance
        wasserstein_variance = sum(
            list(map(
                lambda x, y: self.get_wasserstein_distance(x, y), self.current_eval_visit_prob_list,
                itertools.repeat(normalized_current_policy_p, len(self.current_eval_visit_prob_list))
            ))
        ) / (self.eval_count - 1)


        self.cumulative_visited_sectors_list = [
            self.cumulative_visited_sectors_list[i] + self.current_eval_visited_sectors_list[i] for i in
            range(self.num_sectors)]

        self.current_eval_visited_sectors_list = [0 for _ in range(self.num_sectors)]
        self.eval_count = 0
        self.current_eval_visit_prob_list = []

        return policy_score, policy_gain, cross_entropy, total_variance, wasserstein_variance


def arr_to_sig(arr):
    """Convert a 2D array to a signature for cv2.EMD"""

    # cv2.EMD requires single-precision, floating-point input
    sig = np.empty((arr.size, 3), dtype=np.float32)
    count = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            sig[count] = np.array([arr[i, j], i, j])
            count += 1
    return sig
