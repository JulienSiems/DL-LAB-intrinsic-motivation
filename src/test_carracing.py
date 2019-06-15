from __future__ import print_function

import torch
import gym
from agent.dqn_agent import DQNAgent
from train_carracing import run_episode
from agent.networks import *
import numpy as np
import os
from datetime import datetime
import json

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped
    env = wrappers.Monitor(env, './', force=True)

    history_length = 2
    state_dim = (history_length, 96, 96)
    num_actions = 5
    batch_size = 16

    # Define networks and load agent
    qnet = CNN(history_length=history_length, n_classes=num_actions)
    qtarget = CNN(history_length=history_length, n_classes=num_actions)
    agent = DQNAgent(Q=qnet, Q_target=qtarget, num_actions=num_actions, gamma=0.95, batch_size=batch_size, epsilon=0.1,
                     tau=0.01, lr=1e-4, state_dim=state_dim)

    model_dir = "./models_carracing"
    file_name = "dqn_agent_hist2_b32_vc50_e.pt"
    checkpoint_data = agent.load_best_configuration(os.path.join(model_dir, file_name))
    # checkpoint_data = agent.load(os.path.join(model_dir, file_name))

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True, history_length=history_length)
        episode_rewards.append(stats.episode_reward)
        print("Episode: {} Reward: {}".format(i, stats.episode_reward))

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

