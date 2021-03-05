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

def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)