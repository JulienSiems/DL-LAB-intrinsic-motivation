from __future__ import print_function

import argparse
import numpy as np
import pickle
import os
from datetime import datetime
import gzip
import json
import vizdoom_env as vsd

DOOM_MAPS = [
    'my_way_home_org',
    'my_way_home_spwnhard',
    'my_way_home_spwnhard_nogoal',
]


def store_data(data, datasets_dir="./data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file,'wb')
    pickle.dump(data, f)


def save_results(episode_rewards, results_dir="./results"):
    # save results
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

     # save statistics in a dictionary and write them into a .json file
    results = dict()
    results["number_episodes"] = len(episode_rewards)
    results["episode_rewards"] = episode_rewards

    results["mean_all_episodes"] = np.array(episode_rewards).mean()
    results["std_all_episodes"] = np.array(episode_rewards).std()
 
    fname = os.path.join(results_dir, "results_manually-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))
    fh = open(fname, "w")
    json.dump(results, fh)
    print('... finished')


buttons = [

]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--collect_data", action="store_true", default=False, help="Collect the data in a pickle file.")

    args = parser.parse_args()

    samples = {
        "state": [],
        "next_state": [],
        "reward": [],
        "action": [],
        "terminal" : [],
    }

    env = vsd.DoomEnv(map_name=DOOM_MAPS[1], render=True, play=True)
    a = env.num_actions
    episode_rewards = []
    steps = 0
    while True:
        episode_reward = 0
        state = env.reset()
        states = []
        actions = []
        next_states = []
        rewards = []
        terminals = []
        while True:
            next_state, r, done, info = env.step(a)
            episode_reward += r
            a = env.current_action()

            states.append(state)
            actions.append(np.array(a))
            next_states.append(next_state)
            rewards.append(r)
            terminals.append(done)
            
            state = next_state
            steps += 1

            if done:
                print('Next episode')
                break



    env.close()

    

   
