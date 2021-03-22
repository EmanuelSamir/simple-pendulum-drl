import sys
sys.path.insert(0, '../')

from algorithms.a2c import A2CAgent
import gym_robot2d
#from robot2d import Robot2D
from torch import nn
import torch
import gym
import time
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('Acrobot-v1')

# Environments parameters
s = env.reset()

state_dim = s.shape[0]
# actions =       [
#                     np.array([0]),
#                     np.array([-1]),
#                     np.array([1]),
#                     np.array([2]),
#                     np.array([-2])
#                 ]
action_dim = env.action_space.n
actions = None

agent = A2CAgent(env, state_dim, action_dim, actions, 20)

agent.train()  

agent.test()

agent.actor_logger.plot_reward(show = True, save = True)