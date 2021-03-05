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
        mem_capacity = 10_000, 
        n_episodes = 1_000, 
        gamma = 0.999, 
        eps_max = 1.0, 
        eps_min = 0.1, 
        load_model_path = None,
        load_target_model_path = None):


        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.eps = lambda x: max(0.1, eps_max - x*(eps_max-eps_min)/n_episodes)

        # Discrete actions
        self.actions = actions

        # Models
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.load_models(load_model_path, load_target_model_path)

        # Optimizers
        self.model_optimizer = torch.optim.Adam(self.model.parameters())

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
        try:
            for episode in range(self.n_episodes):
                # Reset environment
                state = self.env.reset()
                is_done = False

                # Reset
                episode_reward = 0
                episode_model_loss = 0
                episode_target_model_loss = 0
                

                while not is_done:
                    # Action sampling
                    if random.random() < self.eps(episode):
                        # Exploration
                        action = random.sample(range(len(self.actions)), 1)[0] 
                    else:
                        with torch.no_grad():
                            # Exploitation. Feeding model. Chosen max
                            action = self.model(t(state)).max(0)[1].item()

                    # Update env
                    next_state, reward, is_done, _ = self.env.step(np.array([self.actions[action]]))

                    # Memory update
                    self.memory.update(torch.from_numpy(state), 
                                torch.tensor([action]), 
                                torch.tensor([reward]).float(),  
                                torch.from_numpy(next_state))

                    # Sampling s,a,r,s'
                    states, actions, rewards, next_states  = self.memory.sample(sample_size)

                    # Update model
                    model_loss, target_model_loss = self.update_models(states, actions, rewards, next_states)

                    # Record losses and reward
                    episode_model_loss += model_loss
                    episode_target_model_loss += target_model_loss
                    episode_reward += reward

                    state = next_state
                
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
            action = random.sample(range(len(self.actions)), 1)[0]
            next_state, reward, is_done, _ = self.env.step(np.array([self.actions[action]]))
            self.memory.update( torch.from_numpy(state), 
                                torch.tensor([action]), 
                                torch.tensor([reward]).float(),  
                                torch.from_numpy(next_state))
            state = next_state

    def update_models(self, states, actions, rewards, next_states):
        max_q = self.target_model(next_states).max(1)[0].detach().unsqueeze(1)
        
        y = rewards + max_q * self.gamma

        # Get Q for every action
        q = self.model(states).gather(1, actions)

        # Update model
        model_loss = F.smooth_l1_loss(q, y)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        # Update target model. Pyonik method
        soft_update(self.model, self.target_model, tau = 0.999)
        
        return float(model_loss), float(0)

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False

        while not is_done:
            # Feed Q-network
            action = self.model(t(state)).max(0)[1].item()

            # Choose action
            next_state, reward, is_done, _ = self.env.step(np.array([self.actions[action]]))

            state = next_state
            time.sleep(0.01)
            self.env.render()

        self.env.close()

