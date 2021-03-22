from torch import nn
import torch
import gym
import time
import numpy as np
from models.dqn import DQN
from tqdm import tqdm
from src.utils import *
from src.logger import Logger
from src.memory import ReplayMemory, Transition
import random
import torch.nn.functional as F


class DQNAgent:
    def __init__(
        self, env, state_dim, action_dim, actions, 
        mem_capacity = 1_000, 
        n_episodes = 10_000, 
        gamma = 0.9999, 
        eps_max = 1.0, 
        eps_min = 0.1, 
        load_model_path = None,
        load_target_model_path = None):

        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.eps = lambda x: max(0.09, eps_max - x*(eps_max-eps_min)/n_episodes*1.8)

        # Discrete actions
        self.actions = actions
        self.action_dim = action_dim

        # Models
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.load_models(load_model_path, load_target_model_path)

        # Optimizers
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-3) # best one 5e-4, 9e-4

        # Loggers
        self.model_logger = Logger("DQN", "model")
        self.target_model_logger = Logger("DQN", "target_model")
        self.memory = ReplayMemory(mem_capacity)
        

    def load_models(self, model_path = None, target_model_path = None):
        # model loading
        if model_path:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # target_model loading
        if target_model_path:
            checkpoint = torch.load(target_model_path)
            self.target_model.load_state_dict(checkpoint["model_state_dict"])

    def train(self, sample_size = 32):
        self.fill_memory(sample_size)

        pbar = tqdm(total=self.n_episodes, position=0, leave=True)
        max_steps = 0
        try:
            for episode in range(self.n_episodes):
                # Reset environment
                state = self.env.reset() 
                is_done = False

                # Reset
                episode_reward = 0
                episode_model_loss = 0
                episode_target_model_loss = 0
                
                steps = 0

                while not is_done:
                    steps += 1
                    # Action sampling
                    if random.random() < self.eps(episode):
                        # Exploration
                        action = random.sample(range(self.action_dim), 1)[0] 
                    else:
                        with torch.no_grad():
                            # Exploitation. Feeding model. Chosen max
                            action = self.model(t(state)).max(0)[1].item()

                    # Update env
                    if self.actions:
                        next_state, reward, is_done, _ = self.env.step(np.array(self.actions[action]))
                    else:
                        next_state, reward, is_done, _ = self.env.step(action)

                    if is_done:
                        reward = float(0)

                    # Memory update
                    if (steps == 500):
                        self.memory.update(
                                preprocess_transition(state), 
                                preprocess_transition(action), 
                                preprocess_transition(reward),  
                                preprocess_transition(next_state),
                                preprocess_transition(float(True))
                                )
                    else:
                        self.memory.update(
                                preprocess_transition(state), 
                                preprocess_transition(action), 
                                preprocess_transition(reward),  
                                preprocess_transition(next_state),
                                preprocess_transition(float(is_done))
                                )

                    # Sampling s,a,r,s'
                    states, actions, rewards, next_states, is_dones  = self.memory.sample(sample_size)

                    # Update model
                    model_loss, target_model_loss = self.update_models(states, actions, rewards, next_states, is_dones)

                    # Record losses and reward
                    episode_model_loss += model_loss
                    episode_target_model_loss += target_model_loss
                    episode_reward += reward

                    state = next_state

                    if (max_steps < steps):
                        max_steps = steps
                        #print(max_steps)
                
                self.model_logger.update(episode_model_loss, episode_reward, self.model)
                self.target_model_logger.update(episode_target_model_loss, episode_reward, self.target_model)
                pbar.update()

        except KeyboardInterrupt:
            print("Out because iterruption by user")

        finally:
            try:
                self.model_logger.exception_arisen(self.model)
                self.target_model_logger.exception_arisen(self.target_model)
            except:
                pass
        pbar.close()

    def fill_memory(self, sample_size):
        state = self.env.reset()
        for _ in range(sample_size):
            action = random.sample(range(self.action_dim), 1)[0]
            if self.actions:
                next_state, reward, is_done, _ = self.env.step(np.array([self.actions[action]]))
            else:
                next_state, reward, is_done, _ = self.env.step(action)
            self.memory.update(
                                preprocess_transition(state), 
                                preprocess_transition(action), 
                                preprocess_transition(reward),  
                                preprocess_transition(next_state),
                                preprocess_transition(float(is_done))
                                )
            state = next_state

    def update_models(self, states, actions, rewards, next_states, is_dones):
        max_q = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)
        x = nn.BatchNorm1d(1, affine = False)
        # Check why a 32x32 was better. WHYyYYYY
        max_q_norm = x(max_q.reshape(-1,1))
        y = rewards + max_q_norm * self.gamma * (1 - is_dones)

        # Get Q for every action
        q = self.model(states).gather(1, actions)

        # Update model
        model_loss = F.smooth_l1_loss(q, y)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        
        # print(model_loss.data)
        self.model_optimizer.step()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)

        # Update target model. Pyonik method
        soft_update(self.model, self.target_model, tau = 0.99)
        
        return float(model_loss.data), float(0)

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False

        while not is_done:
            # Feed Q-network
            action = self.model(t(state)).max(0)[1].item()

            # Choose action
            if self.actions:
                next_state, reward, is_done, _ = self.env.step(np.array(self.actions[action]))
            else:
                next_state, reward, is_done, _ = self.env.step(action)

            state = next_state
            time.sleep(0.02)
            self.env.render()

        self.env.close()

