import os
import sys

sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

import numpy as np
from tkinter import filedialog
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def validate(env_class):
    
    # select ppo model
    experiments_path = os.path.join(
        os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir),
        "training",
        "experiments"
    )
    model_path = filedialog.askopenfilename(initialdir=experiments_path)
    norm_stats_path = os.path.normpath(model_path + os.sep + os.pardir)

    # load ppo model
    model = PPO.load(
        path=model_path,
        device='auto'
    )

    env = env_class(
        norm_stats_path=norm_stats_path,
        training=False
    )

    observation, info = env.reset()
    terminated = False

    while not terminated:

        action, _states = model.predict(observation)
        observation, rewards, terminated, truncated, info = env.step(action)

    env.plot_results()


if __name__ == "__main__":

    from envs.attitude_control.environment import AttitudeControlEnv

    validate(env_class=AttitudeControlEnv)
