from torch import nn
import torch
import gym
import time
import numpy as np
from models.a2c import Actor, Critic
from tqdm import tqdm
from src.utils import *


class A2CAgent:
    def __init__(self, env, state_dim, action_dim, actions, n_episodes = 1_000, gamma = 0.999):
        self.env = env
        self.n_episodes = n_episodes
        self.episode_rewards = []
        self.losses = []
        self.gamma = gamma

        # Discrete actions
        self.actions = actions

        # Models
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())


    def train(self):
        pbar = tqdm(total=self.n_episodes, position=0, leave=True)
        for episode in range(self.n_episodes):
            # Reset environment
            state = self.env.reset()
            is_done = False

            # Reset
            total_reward = 0
            advantage_total = []
            
            while not is_done:
                # Feed Policy network
                probs = self.actor(t(state))

                # Choose sample accoding to policy
                action_dist = torch.distributions.Categorical(probs = probs)
                action = action_dist.sample()
                action_ix = action.detach().data.numpy()

                # Update env
                next_state, reward, is_done, info = self.env.step(self.actions[action_ix])

                # Advantage 
                advantage = reward + (1-is_done)* self.gamma * self.critic(t(next_state)) - self.critic(t(state))
                
                critic_loss, actor_loss = self.update_models(advantage, action_dist, action)

                self.losses.append([actor_loss, critic_loss])
                total_reward += reward
                state = next_state
            
            self.episode_rewards.append(total_reward)
            pbar.update()

    def update_models(self, advantage, action_dist, action):
        # Critic update
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = - action_dist.log_prob(action) * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss, actor_loss

    def test(self):
        # Reset environment
        state = self.env.reset()
        is_done = False

        while not is_done:
            # Feed Policy network
            probs = self.actor(t(state))

            # Choose sample accoding to policy
            action_dist = torch.distributions.Categorical(probs = probs)
            action = action_dist.sample()
            action_ix = action.detach().data.numpy()

            # Update env
            next_state, reward, is_done, info = self.env.step(self.actions[action_ix])

            state = next_state
            time.sleep(0.01)
            self.env.render()

        self.env.close()

