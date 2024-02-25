import os
import json
import numpy as np
from gymnasium import spaces
from .running_mean_std import RunningMeanStd


class Normalizator():
    def __init__(
        self,
        obs_space_shape,
        norm_stats_path=None,
        norm_obs=True,
        norm_reward=True,
        training=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8,
    ):

        self.obs_space_shape = obs_space_shape
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.training = training
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        self.ret_rms = RunningMeanStd(shape=())
        self.obs_rms = RunningMeanStd(shape=obs_space_shape)
        self.returns = np.zeros(1)

        if isinstance(norm_stats_path, str):
            self.load_stats(norm_stats_path)

    def init_obs_space(self):

        observation_space = spaces.Box(
            low=-self.clip_obs,
            high=self.clip_obs,
            shape=self.obs_space_shape,
            dtype=np.float32,
        )

        return observation_space

    def reset(self):

        self.returns = np.zeros(1)

    def get_stats(self):

        norm_stats = {
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "obs_rms_mean": self.obs_rms.mean.tolist(),
            "obs_rms_var": self.obs_rms.var.tolist(),
            "obs_rms_count": self.obs_rms.count,
            "ret_rms_mean": self.ret_rms.mean.tolist(),
            "ret_rms_var": self.ret_rms.var.tolist(),
            "ret_rms_count": self.ret_rms.count
        }

        return norm_stats

    def get_normalized_obs(self, obs):
        
        if self.training:
            self._update_obs_stats(obs)
        
        return self._normalize_obs(obs)

    def get_normalized_reward(self, reward):
        
        if self.training:
            self._normalize_rewards(reward)
        
        return self._normalize_rewards(reward)

    def _update_obs_stats(self, obs):

        # update observation normalization statistics
        self.obs_rms.update(obs)

    def _normalize_obs(self, obs):

        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _update_rewards_stats(self, reward):

        # update reward normalization statistics
        self.returns = self.returns * self.gamma + reward
        self.ret_rms.update(self.returns)

    def _normalize_rewards(self, reward):

        return np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)

    def load_stats(self, path):
        
        file_path = os.path.join(path, "normalizator_stats.json")

        with open(file_path, "r") as json_file:
            norm_stats = json.load(json_file)

        self.clip_obs = norm_stats["clip_obs"]
        self.clip_reward = norm_stats["clip_reward"]
        self.gamma = norm_stats["gamma"]
        self.epsilon = norm_stats["epsilon"]
        self.obs_rms.mean = np.array(norm_stats["obs_rms_mean"], np.float64)
        self.obs_rms.var = np.array(norm_stats["obs_rms_var"], np.float64)
        self.obs_rms.count = norm_stats["obs_rms_count"]
        self.ret_rms.mean = np.array(norm_stats["ret_rms_mean"], np.float64)
        self.ret_rms.var = np.array(norm_stats["ret_rms_var"], np.float64)
        self.ret_rms.count = norm_stats["ret_rms_count"]

    def save_stats(self, path):
        
        norm_stats = {
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "obs_rms_mean": self.obs_rms.mean.tolist(),
            "obs_rms_var": self.obs_rms.var.tolist(),
            "obs_rms_count": self.obs_rms.count,
            "ret_rms_mean": self.ret_rms.mean.tolist(),
            "ret_rms_var": self.ret_rms.var.tolist(),
            "ret_rms_count": self.ret_rms.count
        }

        file_path = os.path.join(path, "normalizator_stats.json")

        with open(file_path, "w") as json_file:
            json.dump(norm_stats, json_file)




if __name__ == '__main__':

    output_filepath = os.path.dirname(__file__)

    norm = Normalizator(7)

    obs = np.array([1, 0, 0, 0, 10, -30, 0])
    reward = 12
    for i in range(100):
        obs_ = norm.get_normalized_obs(obs)
        reward_ = norm.get_normalized_reward(reward)
        print(obs_, reward_)



    # output_filepath = os.path.dirname(__file__)

    # norm.save_stats(output_filepath)