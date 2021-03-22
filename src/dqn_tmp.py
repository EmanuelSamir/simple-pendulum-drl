import sys
sys.path.insert(0, '../')

from algorithms.dqn import DQNAgent
from torch import nn
import torch
import gym
import time
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('MountainCarContinuous-v0')

# Environments parameters
state_dim = env.observation_space.shape[0]

action_dim = 5 #env.action_space.n
actions =       [
                    np.array([0]),
                    np.array([-1]),
                    np.array([1]),
                    np.array([2]),
                    np.array([-2])
                ]


agent = DQNAgent(env, state_dim, action_dim, actions, n_episodes = 500, load_model_path='../checkpoints/DQN/target_model/tmp_model.pth', load_target_model_path='../checkpoints/DQN/model/tmp_model.pth')

#agent.train()  

agent.test()

#agent.model_logger.plot_reward(show = True, save = True)
#agent.model_logger.plot_loss(show = True, save = True)