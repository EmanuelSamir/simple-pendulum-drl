import sys
sys.path.insert(0, '../')

from algorithms.dqn import DQNAgent
from torch import nn
import torch
import gym
import time
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("Pendulum-v0")

# Environments parameters
state_dim = env.observation_space.shape[0]
actions = np.array([-2.2,-0.6,0.,0.6,2.2])
action_dim = actions.shape[0]

agent = DQNAgent(env, state_dim, action_dim, actions, n_episodes = 100)

agent.train()  

agent.test()

agent.model_logger.plot_reward(show = True, save = True)