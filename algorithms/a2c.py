from torch import nn
import torch
import gym
import time
import numpy as np
from models.a2c import Actor, Critic
from tqdm import tqdm
from src.utils import *
from src.logger import Logger


class A2CAgent:
    def __init__(self, env, state_dim, action_dim, actions, 
                n_episodes = 1_000, 
                gamma = 0.999,
                load_actor_path = None,
                load_critic_path = None):
        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma

        # Discrete actions
        self.actions = actions

        # Models
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.load_models(load_actor_path, load_critic_path)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        # Loggers
        self.actor_logger = Logger("A2C", "actor")
        self.critic_logger = Logger("A2C", "critic")

    def load_models(self, actor_path = None, critic_path = None):
        # Actor loading
        if actor_path:
            checkpoint = torch.load(actor_path)
            self.actor.load_state_dict(checkpoint["model_state_dict"])

        # Critic loading
        if critic_path:
            checkpoint = torch.load(critic_path)
            self.critic.load_state_dict(checkpoint["model_state_dict"])

    def train(self):
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

                advantage_total = []                
                
                while not is_done:
                    # Feed Policy network
                    probs = self.actor(t(state))

                    # Choose sample accoding to policy
                    action_dist = torch.distributions.Categorical(probs = probs)
                    action = action_dist.sample()
                    action_ix = action.detach().data.numpy()

                    # Update env
                    if self.actions:
                        next_state, reward, is_done, info = self.env.step(self.actions[action_ix])
                    else:
                        next_state, reward, is_done, info = self.env.step(action_ix)

                    
                    # Advantage 
                    advantage = reward + (1-is_done)* self.gamma * self.critic(t(next_state)) - self.critic(t(state))
                    
                    # Update models
                    critic_loss, actor_loss = self.update_models(advantage, action_dist, action)

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
            self.actor_logger.exception_arisen(self.actor)
            try:
                self.critic_logger.exception_arisen(self.critic)
            except:
                pass
        pbar.close()

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
        
        return float(critic_loss), float(actor_loss)

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

            if self.actions:
                next_state, reward, is_done, info = self.env.step(self.actions[action_ix])
            else:
                next_state, reward, is_done, info = self.env.step(action_ix)

            state = next_state
            time.sleep(0.01)
            self.env.render()

        self.env.close()

