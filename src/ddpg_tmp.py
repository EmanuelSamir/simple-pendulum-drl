import sys
sys.path.insert(0, '../')

from algorithms.ddpg import DDPGAgent
from torch import nn
import torch
import gym
import time
import numpy as np
from matplotlib import pyplot as plt

env = gym.make("MountainCarContinuous-v0")

# Environments parameters
state_dim = env.observation_space.shape[0]

# No actions given. Continuous case
action_dim = 1

agent = DDPGAgent(env, state_dim, action_dim, n_episodes=100)

agent.train()  

agent.test()

agent.actor_logger.plot_reward(show = True, save = True)
agent.actor_logger.plot_loss(show = True, save = True)