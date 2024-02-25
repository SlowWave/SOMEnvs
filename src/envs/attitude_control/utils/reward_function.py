import os
import tomli
import numpy as np


# get config data
with open(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
            "configs",
            "config.toml"
        ),
        "rb"
) as config_file:
    CFG = tomli.load(config_file)


class RewardFunctionModel():
    def __init__(self):

        # define mapping dictionary
        self.reward_model_map = {
            "1": self._model_1,
            "2": self._model_2,
        }

        self.model_id = str(CFG["gymnasium"]["reward_function"]["model_id"])

    def get_reward(self, storage):

        return self.reward_model_map[self.model_id](storage)

    def _model_1(self, storage):

        if storage.angular_attitude_error[-1] < storage.angular_attitude_error[-2]:
            reward_1 = 1
        else:
            reward_1 = -1

        is_last_reward = False
        rewards = [reward_1]

        return is_last_reward, rewards

    def _model_2(slef, storage):

        r1 = 0
        r2 = 3
        r3 = -0.5

        is_last_reward = False

        rewards = [r1, r2, r3]

        return is_last_reward, rewards

