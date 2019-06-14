import inspect
import json

import numpy as np
import os
from tensorboardX import SummaryWriter

from datetime import datetime
import socket

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE = 3
BRAKE = 4


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

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))
