import sys
sys.path.insert(0, '../')

from algorithms.dqn import DQNAgent
from torch import nn
import torch
import gym
import time
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('CartPole-v1')

# Environments parameters
state_dim = env.observation_space.shape[0]
actions = None
action_dim = 2


agent = DQNAgent(env, state_dim, action_dim, actions, n_episodes = 1000)

agent.train()  

agent.test()

agent.model_logger.plot_reward(show = True, save = True)
agent.model_logger.plot_loss(show = True, save = True)