import os
import sys

sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

import subprocess
import numpy as np
from utils.tb_logger import TensorboardCallback
from utils.venv_config_dump import VenvConfigDumpCallback
from utils.tb_check import is_tensorboard_running, close_tensorboard_processes
from tkinter import filedialog
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def train(
        env_class,
        resume_training=False,
        training_timesteps=1000,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        n_envs=4,
        experiment_info_str="",
        policy_kwargs=None,
        run_tensorboard=False,
):

    # get model and norm_stats paths
    if resume_training:
        experiments_path = os.path.join(
            os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir),
            "training",
            "experiments"
        )
        model_path = filedialog.askopenfilename(initialdir=experiments_path)
        norm_stats_path = os.path.normpath(model_path + os.sep + os.pardir)
    else:
        norm_stats_path = None

    # define output path
    datetime_str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    model_str = "PPO" + experiment_info_str
    file_name = model_str + "_" + datetime_str
    output_filepath = os.path.join(os.path.dirname(__file__), "experiments", file_name)
    tensorboard_log_path = os.path.join(output_filepath, "tensorboard_data")
    os.mkdir(output_filepath)

    # define vectorized environment(s)
    venv = make_vec_env(
        env_id=env_class,
        n_envs=n_envs,
        seed=np.random.randint(0, 2**31 - 1),
        env_kwargs={"norm_stats_path":norm_stats_path, "training":True},
    )

    # define ppo model
    if resume_training:
        model = PPO.load(
            path=model_path,
            env=venv,
            device="auto",
            tensorboard_log=tensorboard_log_path,
        )
    else:
        model = PPO(
            policy="MlpPolicy",
            env=venv,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=None,
            stats_window_size=100,
            tensorboard_log=tensorboard_log_path,
            policy_kwargs=policy_kwargs,
            device="auto",
        )

    # run tensorboard
    if run_tensorboard:
        if is_tensorboard_running():
            close_tensorboard_processes()
        subprocess.Popen(
            ["tensorboard", "--logdir", tensorboard_log_path]
        )

    # train ppo agent
    model.learn(
        total_timesteps=training_timesteps,
        callback=[
            TensorboardCallback(),
            VenvConfigDumpCallback(output_filepath),
            # LearningRateCallback(initial_lr=1e-4, final_lr=1e-6, total_epochs=20)
        ],
        reset_num_timesteps=False if resume_training else True,
        progress_bar=True,
    )

    # save training data
    model.save(os.path.join(output_filepath, "agent_model"))


if __name__ == "__main__":

    from envs.attitude_control.environment import AttitudeControlEnv

    train(
        env_class=AttitudeControlEnv,
        training_timesteps=1000,
        n_envs=1,
        resume_training=False,
        run_tensorboard=False
    )
