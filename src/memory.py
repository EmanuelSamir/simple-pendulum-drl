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
        # num (float) -> torch 1 dim -> num (float) (dim, sample) 1, 32
        # array -> torch 2 dim                       2, 32  
        #for i in range(32):
        rewards = torch.cat(batch.reward).float()
        next_states = torch.cat(batch.next_state).float()
        states = torch.cat(batch.state).float()
        actions = torch.cat(batch.action)
        is_dones = torch.cat(batch.is_done)
        return states, actions, rewards, next_states, is_dones
    