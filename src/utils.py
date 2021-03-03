from torch import nn
import torch
import gym
import time
import numpy as np
import os

def t(x): return torch.from_numpy(x).float()

def create_dir(save_path):
    path = ""
    for directory in os.path.split(save_path):
        path = os.path.join(path, directory)
        if not os.path.exists(path):
            os.mkdir(path)