import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4
N_ACTIONS = 5

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.empty((rgb.shape[0], rgb.shape[1], rgb.shape[2]))
    batch_size = 64
    N = rgb.shape[0]
    for i in range(0, N, batch_size):
        if i + batch_size - 1 < N:
            input = rgb[i: i + batch_size] / 255.0
            gray[i: i + batch_size] = np.dot(input[..., :3], [0.2125, 0.7154, 0.0721])
            gray[..., -12:, :] = 0.0
        else:
            input = rgb[i:N] / 255.0
            gray[i:N] = np.dot(input[..., :3], [0.2125, 0.7154, 0.0721])
            gray[..., -12:, :] = 0.0
    return gray.astype('float32')

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(press_lr_acc=0, press_lr_br=0)
def action_to_id(a):
    """ action_to_id
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if a[0] != 0 and a[1] != 0:
        if action_to_id.press_lr_acc % 2 == 0:
            a[0] = 0
        else:
            a[1] = 0
        action_to_id.press_lr_acc += 1
    if a[0] != 0 and a[2] != 0:
        if action_to_id.press_lr_br % 2 == 0:
            a[0] = 0
        else:
            a[2] = 0
        action_to_id.press_lr_br += 1
    if a[1] != 0 and a[2] != 0:
        a[1] = 0

    if np.allclose(a, [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif np.allclose(a, [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif np.allclose(a, [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif np.allclose(a, [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    else:
        return STRAIGHT                                      # STRAIGHT: 0

def id_to_action(action_id, max_speed=0.8):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
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
        self.episode_loss = 0

    def step(self, reward, action_id, loss):
        self.episode_reward += reward
        self.actions_ids.append(action_id)
        self.episode_loss += loss

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))
