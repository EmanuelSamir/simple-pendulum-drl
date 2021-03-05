from torch import nn
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(state_dim, 32),
                    nn.Sigmoid(),
                    nn.Linear(32,8),
                    nn.Sigmoid(),
                    nn.Linear(8,action_dim),
                    )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(state_dim + action_dim, 32),
                    nn.Sigmoid(),
                    nn.Linear(32,8),
                    nn.Sigmoid(),
                    nn.Linear(8,1)
                    )
    def forward(self, s, a):        
        return self.net(torch.cat((s,a), dim = 1))