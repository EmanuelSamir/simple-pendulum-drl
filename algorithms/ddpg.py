from torch import nn
import torch
import gym
import time
import numpy as np
from models.ddpg import Actor, Critic
from tqdm import tqdm
from src.utils import *
from src.logger import Logger
from src.memory import ReplayMemory, Transition
import random
import torch.nn.functional as F


class DDPGAgent:
    def __init__(
        self, env, state_dim, action_dim,
        mem_capacity = 10_000, 
        n_episodes = 1_000, 
        gamma = 0.999, 
        eps_max = 1.0, 
        eps_min = 0.1, 
        load_actor_path = None,
        load_critic_path = None):


        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma

        # Models
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.load_models(load_actor_path, load_critic_path)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 5e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 5e-3)

        # Loggers
        self.actor_logger = Logger("DDPG", "actor")
        self.critic_logger = Logger("DDPG", "critic")

        self.memory = ReplayMemory(mem_capacity)
        

    def load_models(self, actor_path = None, critic_path = None):
        # actor loading
        if actor_path:
            checkpoint = torch.load(actor_path)
            self.actor.load_state_dict(checkpoint["model_state_dict"])
            self.actor_target.load_state_dict(checkpoint["model_state_dict"])

        # critic loading
        if critic_path:
            checkpoint = torch.load(critic_path)
            self.critic.load_state_dict(checkpoint["model_state_dict"])
            self.critic_target.load_state_dict(checkpoint["model_state_dict"])

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
                episode_actor_loss = 0
                episode_critic_loss = 0
                

                while not is_done:
                    # Action sampling
                    action = self.actor(t(state)).detach().float()

                    # Update env
                    next_state, reward, is_done, _ = self.env.step(action)

                    # Memory update
                    self.memory.update(
                                preprocess_transition(state, force_to_float= True), 
                                preprocess_transition(action), 
                                preprocess_transition(reward),  
                                preprocess_transition(next_state, force_to_float= True),
                                preprocess_transition(float(is_done))
                                )

                    # Sampling s,a,r,s'
                    states, actions, rewards, next_states, is_dones  = self.memory.sample(sample_size)

                    # Update model
                    actor_loss, critic_loss = self.update_models(states, actions, rewards, next_states, is_dones)

                    # Record losses and reward
                    episode_actor_loss += actor_loss
                    episode_critic_loss += critic_loss
                    episode_reward += reward

                    state = next_state
                
                self.actor_logger.update(episode_actor_loss, episode_reward, self.actor)
                self.critic_logger.update(episode_critic_loss, episode_reward, self.critic)
                pbar.update()

        except KeyboardInterrupt:
            print("Out because iterruption by user")

        finally:
            try:
                self.actor_logger.exception_arisen(self.actor)
                self.critic_logger.exception_arisen(self.critic)
            except:
                pass
        pbar.close()

    def fill_memory(self, sample_size):
        state = self.env.reset()
        for _ in range(sample_size):
            action = self.actor(t(state)).detach().float()
            next_state, reward, is_done, _ = self.env.step(action)
            self.memory.update(
                                preprocess_transition(state, force_to_float= True), 
                                preprocess_transition(action), 
                                preprocess_transition(reward),  
                                preprocess_transition(next_state, force_to_float= True),
                                preprocess_transition(float(is_done))
                                )
            state = next_state

    def update_models(self, states, actions, rewards, next_states, is_dones):
        additional_qs = self.critic_target(next_states, self.actor_target(next_states))
        x = nn.BatchNorm1d(1, affine = False)

        # Check why a 32x32 was better. WHYyYYYY
        max_qs_norm = x(additional_qs.reshape(-1,1))
        
        y = rewards + max_qs_norm * self.gamma * is_dones

        # Get Q for every action and state
        q = self.critic(states, actions)

        # Update critic
        loss_critic = F.smooth_l1_loss(q, y)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        for param in self.critic.parameters():
            param.grad.data.clamp_(-1, 1)
        self.critic_optimizer.step()

        # Update agent
        loss_actor = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()
        

        # Update target models. Pyonik method
        soft_update(self.actor, self.actor_target, 0.99)
        soft_update(self.critic, self.critic_target, 0.99)
        
        return loss_critic.item(), loss_actor.item()

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False

        while not is_done:
            # Actor output
            action = self.actor(t(state)).detach().numpy()

            # Choose action
            next_state, reward, is_done, _ = self.env.step(action)

            state = next_state
            time.sleep(0.01)
            self.env.render()

        self.env.close()

