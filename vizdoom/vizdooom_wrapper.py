import os
import sys
import numpy as np
from vizdoom import *

DOOM_MAPS = [
    'my_way_home_org',
    'my_way_home_spwnhard',
    'my_way_home_spwnhard_nogoal',
]


class DoomEnv(object):

    def __init__(self, map_name=None, seed=None):
        self.map_name = map_name

        # init game from vizdoom
        self.game = DoomGame()
        # self.game.set_seed(seed)
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, self.map_name + ".cfg"))
        # set modified configs here
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_window_visible(False)

        self.game.init()

        # init actions
        self._init_one_hot_actions()

    def _init_one_hot_actions(self):
        self.action_oh = []
        num_action = len(self.game.get_available_buttons())
        for i in range(num_action):
            zeros = [0] * num_action
            zeros[i] = 1
            self.action_oh.append(zeros)

    def _action2hot(self, action):
        try:
            return self.action_oh[action]
        except:
            raise NotImplementedError

    def reset(self):
        self.game.new_episode()
        self.state = self.game.get_state()
        observation = np.transpose(self.state.screen_buffer, (1, 2, 0))
        reward = None
        done = False

        info = {}
        info['xy_pos'] = (self.game.get_game_variable(GameVariable(37)), self.game.get_game_variable(GameVariable(38)))
        info['rotation'] = self.game.get_game_variable(GameVariable(40))

        return observation, reward, done, info

    def step(self, action, action_repetition=1):
        reward = self.game.make_action(self._action2hot(action), action_repetition)
        done = self.game.is_episode_finished()
        self.state = self.game.get_state()

        # observation = state.screen_buffer
        observation = np.transpose(self.state.screen_buffer, (1, 2, 0))

        info = {}
        info['xy_pos'] = (self.game.get_game_variable(GameVariable(37)), self.game.get_game_variable(GameVariable(38)))
        info['rotation'] = self.game.get_game_variable(GameVariable(40))

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.game.close()

    def seed(self, seed=None):
        pass


if __name__ == '__main__':
    doom = DoomEnv(map_name=DOOM_MAPS[2])
    obs, rew, done, info = doom.step(0)
    obs, rew, done, info = doom.reset()
    import matplotlib.pyplot as plt

    plt.imshow(obs)
    plt.show()
    plt.savefig('test.png')
    print(info)
