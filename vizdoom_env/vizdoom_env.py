import os
import numpy as np
import gym
from vizdoom import *

DOOM_MAPS = [
    'my_way_home_org',
    'my_way_home_spwnhard',
    'my_way_home_spwnhard_nogoal',
]


class DoomEnv(gym.Env):

    def __init__(self, map_name=None, seed=None, render=False, play=False):
        self.map_name = map_name
        # init game from vizdoom
        self.game = DoomGame()
        # self.game.set_seed(seed)
        scenarios_dir = os.path.join(os.path.dirname(__file__), 'scenarios')
        self.game.load_config(os.path.join(scenarios_dir, self.map_name + ".cfg"))
        # set modified configs here
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_window_visible(render)
        if play:
            self.game.set_mode(Mode.SPECTATOR)

        self.num_actions = len(self.game.get_available_buttons())
        self.buttons = self.game.get_available_buttons()
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # init actions
        self._init_one_hot_actions()
        self.game.init()
        self.state = self.game.get_state()


    def _init_one_hot_actions(self):
        self.action_oh = []
        for i in range(self.num_actions):
            zeros = [0] * self.num_actions
            zeros[i] = 1
            self.action_oh.append(zeros)
        self.action_oh.append([0] * self.num_actions)


    def state_image(self):
        return np.transpose(self.state.screen_buffer, (1, 2, 0))

    def _action2hot(self, action):
        try:
            return self.action_oh[action]
        except:
            raise NotImplementedError

    def reset(self):
        if self.state is not None:
            self.game.new_episode()
        self.state = self.game.get_state()
        observation = self.state_image()
        return observation

    def step(self, action, action_repetition=1):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        elif isinstance(action, int):
            action = self._action2hot(action)

        reward = self.game.make_action(action, action_repetition)
        done = self.game.is_episode_finished()
        if not done:
            self.state = self.game.get_state()

        # observation = state.screen_buffer
        observation = self.state_image()

        info = {}
        info['xy_pos'] = (self.game.get_game_variable(GameVariable(37)), self.game.get_game_variable(GameVariable(38)))
        info['rotation'] = self.game.get_game_variable(GameVariable(40))

        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.game.close()

    def current_action(self):
        a = self.num_actions
        for i, b in enumerate(self.game.get_available_buttons()):
            v = self.game.get_button(b)
            if v != 0:
                a = i + 1
                break
        return a

    def seed(self, seed=None):
        pass


if __name__ == '__main__':
    doom = DoomEnv(map_name=DOOM_MAPS[1])
    _, rew, done, info = doom.step(0)
    obs = doom.reset()
    import matplotlib.pyplot as plt

    plt.imshow(obs)
    plt.show()
    print(info)
