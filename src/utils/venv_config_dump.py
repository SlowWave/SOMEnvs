import os
import json
import shutil
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class VenvConfigDumpCallback(BaseCallback):

    def __init__(
        self,
        output_filepath,
        verbose=0
    ):
        
        super(VenvConfigDumpCallback, self).__init__(verbose)
        self.env_obj = None
        self.path = output_filepath

    def _save_env_config(self):

        shutil.copy(
            os.path.join(
                os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir),
                    "envs",
                    "attitude_control",
                    "configs",
                    "config.toml",
                ),
            os.path.join(self.path, "env_config.toml")
        )

    def _save_norm_stats(self):

        norm_stats = {
            "clip_obs": self.env_obj.normalizator.clip_obs,
            "clip_reward": self.env_obj.normalizator.clip_reward,
            "gamma": self.env_obj.normalizator.gamma,
            "epsilon": self.env_obj.normalizator.epsilon,
            "obs_rms_mean": self.env_obj.normalizator.obs_rms.mean.tolist(),
            "obs_rms_var": self.env_obj.normalizator.obs_rms.var.tolist(),
            "obs_rms_count": self.env_obj.normalizator.obs_rms.count,
            "ret_rms_mean": self.env_obj.normalizator.ret_rms.mean.tolist(),
            "ret_rms_var": self.env_obj.normalizator.ret_rms.var.tolist(),
            "ret_rms_count": self.env_obj.normalizator.ret_rms.count
        }

        file_path = os.path.join(self.path, "normalizator_stats.json")

        with open(file_path, "w") as json_file:
            json.dump(norm_stats, json_file)

    def _on_training_end(self):
        
        self.env_obj = self.training_env.envs[0].env
        self._save_env_config()

        if self.env_obj.normalize_observation or self.env_obj.normalize_reward:
            self._save_norm_stats()

    def _on_step(self):
        return True
