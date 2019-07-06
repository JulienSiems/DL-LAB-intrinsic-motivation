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


def setup_experiment_folder_writer(frame, name, log_dir=None, args_for_filename=None):
    """
    Create experiment folder and tensorboardx writer
    :param args_for_filename:
    :param name:
    :param log_dir:
    :param frame: Inspection frame of the main function.
    :return: tensorboardx writer
    """
    args, _, _, values = inspect.getargvalues(frame)
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
        self.actions_ids = []
        self.steps = 0

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)
        self.steps += 1

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)


def plot_trajectory(trajectory, sectors, sector_bb, objects):
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
            'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max, 'area': (x_max - x_min) * (y_max - y_min), 'cog': ((x_max + x_min) / 2, (y_max + y_min) / 2)
        }

    # TODO: Calculate area of non rectangular rooms. Currently fixed for vizdoom.
    sector_bbs['section_16']['area'] = 15360
    sector_bbs['section_16']['cog'] = (488, -640)

    return sector_bbs


# https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


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