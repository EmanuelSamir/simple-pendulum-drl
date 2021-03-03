from torch import nn
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, f1 = 32, f2 = 16):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(state_dim, f1),
                    nn.ReLU(),
                    nn.Linear(f1,f2),
                    nn.ReLU(),
                    nn.Linear(f2,action_dim),
                    nn.Softmax(0)
                    )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, f1 = 32, f2 = 16):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(state_dim, f1),
                    nn.ReLU(),
                    nn.Linear(f1,f2),
                    nn.ReLU(),
                    nn.Linear(f2,1)
                    )
    def forward(self, x):
        return self.net(x)