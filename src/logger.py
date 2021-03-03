
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from torch import save as tsave

from .utils import create_dir


class Logger:
    def __init__(self, algorithm, model):
        self._rewards = []
        self._losses = []
        self.episode = 0
        
        self.save_model_path = os.path.join("models", algorithm, model)
        create_dir(self.save_model_path)
        self.save_data_path = os.path.join("data", algorithm, model)
        create_dir(self.save_data_path)
        self.save_result_path = os.path.join("result", algorithm, model)
        create_dir(self.save_result_path)

        self.best_reward = 0

    def update(self, loss, reward, model, save_best = False, save_checkpoints = True, checkpoint_every = 50):
        
        self._rewards.append(reward)
        self._losses.append(loss)
        self.episode += 1

        if save_best and reward > self.best_reward:
            self.best_reward = reward
            self.save_model(model, "best_model_{}".format(reward))

        if save_checkpoints and self.episode % checkpoint_every == 0:
            self.save_model(model, "e{}_r{}".format(self.episode, reward))

    def report(self):
        losses = self._losses
        rewards = self._rewards
        
        mean_loss = np.mean(losses)
        se_loss = np.std(losses) / np.sqrt(len(losses))

        mean_reward = np.mean(rewards)
        se_reward = np.std(rewards) / np.sqrt(len(rewards))

        print("\nEpisode {}".format(self.episode))
        print("Loss: {:.3f} +/- {:.1f}".format(mean_loss, se_loss))
        print("Reward: {:.3f} +/- {:.1f}".format(mean_reward, se_reward))

    def save_data(self, fn_losses, fn_rewards):
        if not fn_losses.endswith(".pkl"):
            fn_losses += ".pkl"
        if not fn_rewards.endswith(".pkl"):
            fn_rewards += ".pkl"

        with open(os.path.join(self.save_data_path, fn_rewards), "wb") as f:
            pickle.dump(self._rewards, f)
        with open(os.path.join(self.save_data_path, fn_rewards), "wb") as f:
            pickle.dump(self._losses, f)

    def save_model(self, model, fn_model):
        if not fn_name.endswith(".pth"):
            fn_name += ".pth"
        tsave(model, os.path.join(self.save_path, fn_name))

    def exception_arisen(self, model):
        self.report()
        self.save_data("tmp_loss", "tmp_rewards")
        self.save_model(model, "tmp_model")


    def plot_reward(self, sliding_window=50, show=False, save=False):
        rewards = self._moving_average(self._rewards, sliding_window)
        plt.plot(range(len(rewards)), rewards, label=)

        plt.xlabel("Episode")
        plt.ylabel("Total episode reward")
        plt.legend()

        if save:
            plt.savefig(os.path.join(self.save_path, "rewards.png"))

        if show:
            plt.show()

    @staticmethod
    def _moving_average(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, "same")

    
