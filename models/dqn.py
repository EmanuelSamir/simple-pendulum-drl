from torch import nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(state_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32,8),
                    nn.ReLU(),
                    nn.Linear(8,action_dim)
                    )
    def forward(self, x):
        return self.net(x)