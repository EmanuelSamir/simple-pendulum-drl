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


def preprocess_transition(data, force_to_float = False):
    t = torch.tensor(data)
    dim = len(list(t.shape))
    if (dim == 0):
        t = t.unsqueeze(0).unsqueeze(0)
    elif (dim == 1):
        t = t.unsqueeze(0)
    t = t.reshape(1,-1)   
    if force_to_float:
        t = t.float()   
    return t

    # num -> 1, 1 
    # list -> 1 list   len(lst), 1
    # array -> 1, array shape[0], 1
