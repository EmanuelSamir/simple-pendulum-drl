from collections import deque, namedtuple
import random
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'is_done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity 
        self.memory = deque(maxlen = self.capacity)
    
    def update(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        n_batch = random.sample(self.memory, batch_size)

        batch = Transition(*zip(*n_batch))
        rewards = torch.cat(batch.reward).float().unsqueeze(1)
        next_states = torch.stack(batch.next_state).float()
        states = torch.stack(batch.state).float()
        actions = torch.cat(batch.action).unsqueeze(1)
        is_dones = torch.cat(batch.is_done).unsqueeze(1)
        return states, actions, rewards, next_states, is_dones
    