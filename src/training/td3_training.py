import os
import sys

sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

from envs.sadsimenv.environment import SpacecraftEnv
from utils.tb_logger import TensorboardCallback
from utils.venv_config_dump import VenvConfigDumpCallback
import numpy as np
import torch as th
from tqdm import tqdm
from datetime import datetime
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env

# define output path
datetime_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
model_str = "TD3"
file_name = model_str + "_" + datetime_str
output_filepath = os.path.join(os.path.dirname(__file__), "experiments", file_name)
os.mkdir(output_filepath)

# Parallel environments
# The following line is needed to fix a numpy bug, for more info visit: ...
seed_fix = np.random.randint(0, 2**31 - 1)
venv = make_vec_env(
    SpacecraftEnv,
    env_kwargs={"norm_stats_path":None, "training":True},
    seed=seed_fix,
    n_envs=1
)

# define policy params
td3_policy_kwargs = dict(
    net_arch=[400, 300]
)

# define TD3 model
model = TD3(
    "MlpPolicy",
    venv,
    verbose=1,
    device='auto',
    tensorboard_log=os.path.join(output_filepath, "tensorboard_data")
)

# agent training
model.learn(
    total_timesteps=1_00,
    callback=[
        TensorboardCallback(),
        VenvConfigDumpCallback(output_filepath)
        # LearningRateCallback(initial_lr=1e-4, final_lr=1e-6, total_epochs=20)
    ],
    progress_bar=True
)

# save training data
model.save(os.path.join(output_filepath, "agent_model"))

