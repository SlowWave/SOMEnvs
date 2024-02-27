import os
import tomli
import shutil
import gymnasium as gym
from .utils.observation_space import ObservationSpaceModel
from .utils.action_space import ActionSpaceModel
from .utils.reward_function import RewardFunctionModel
from .utils.normalizator import Normalizator
from .utils.storage import Storage
from .basilisk.bsk_simulation_container import BSKSimulationContainer
from .utils.disturbances_generator import DisturbancesGenerator
from .utils.reference_generator import ReferenceGenerator


# get config data
with open(
    os.path.join(
        os.path.dirname(__file__),
            "configs",
            "config.toml"
        ),
        "rb"
) as config_file:
    CFG = tomli.load(config_file)


class AttitudeControlEnv(gym.Env):
    def __init__(self, norm_stats_path=None, training=True):

        # define environment configs
        self.epoch_time_horizon = CFG["gymnasium"]["epoch_time_horizon"]
        self.episode_time_step = 1 / CFG["basilisk"]["fsw_step_frequency"]
        self.use_random_seed = CFG["gymnasium"]["use_random_seed"]
        self.random_seed = CFG["gymnasium"]["random_seed"]
        self.normalize_observation = CFG["gymnasium"]["normalize_observation"]
        self.normalize_reward = CFG["gymnasium"]["normalize_reward"]
        
        # initialize environment components
        self.bsk_simulation = None
        self.disturbances_generator = DisturbancesGenerator()
        self.reference_generator = ReferenceGenerator()
        self.storage = Storage()
        self.observation_space_model = ObservationSpaceModel()
        self.observation_space = self.observation_space_model.get_observation_space()
        self.action_space_model = ActionSpaceModel()
        self.action_space = self.action_space_model.get_action_space()
        self.reward_model = RewardFunctionModel()

        if self.normalize_observation or self.normalize_reward:
            self.normalizator = Normalizator(
                obs_space_shape=self.observation_space.shape,
                norm_stats_path=norm_stats_path,
                norm_obs=self.normalize_observation,
                norm_reward=self.normalize_reward,
                training=training
            )
        else:
            self.normalizator = None

    def reset(self, *, seed=None, options=None):

        # set random seed if needed
        if self.use_random_seed:
            seed = self.random_seed

        # the following line is needed for custom environments
        super().reset(seed=seed)

        # generate torque disturbances 
        torques = self.disturbances_generator.generate_disturbances()

        # generate attitude reference
        reference = self.reference_generator.generate_attitude_reference()

        # reset basilisk simulation
        self.bsk_simulation = BSKSimulationContainer()
        self.bsk_simulation.reset()

        # set torque disturbances and attitude reference
        self.bsk_simulation.set_torque_disturbances(torques)
        self.bsk_simulation.set_attitude_reference(reference)

        # initialize basilisk simulation
        self.bsk_simulation.initialize_simulation()

        # reset normalizator object
        if self.normalize_observation or self.normalize_reward:
            self.normalizator.reset()

        # perform first basilisk simulation step
        self.bsk_simulation.step(self.episode_time_step)
        bsk_states = self.bsk_simulation.get_simulation_data()

        # reset storage object
        self.storage.reset(bsk_states)

        # get first environment observation
        observation = self.storage.get_observation()

        # normalize observation
        if self.normalize_observation:
            observation = self.normalizator.get_normalized_obs(observation)

        info = {}

        return observation, info

    def step(self, action):

        # elaborate action 
        action = self.action_space_model.get_elaborated_action(action, self.storage)

        # perform basilisk simulation step
        self._simulation_step(action)

        # get environmnent observations
        observation = self.storage.get_observation()

        # normalize observation
        if self.normalize_observation:
            observation = self.normalizator.get_normalized_obs(observation)

        # compute agent reward
        is_last_reward, rewards = self.reward_model.get_reward(self.storage)
        reward = sum(rewards)

        # normalize reward
        if self.normalize_reward:
            reward = self.normalizator.get_normalized_reward(reward)

        # check termination condition
        if is_last_reward or self.storage.is_last_step:
            terminated = True
        else:
            terminated = False

        # set truncated parameter to False (unused)
        truncated = False

        # store rewards values so that can be accessed by TensorboardCallback object
        info = {
            'rewards': rewards
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def plot_results(self):

        # plot simulation data
        self.storage.plot_sc_states()
        self.storage.plot_tracking_errors()
        self.storage.plot_control_signals()
        self.storage.plot_rw_torques()
        self.storage.plot_rw_speeds()

    def save_configs(self, path):

        shutil.copy(
            os.path.join(os.path.dirname(__file__), "configs", "config.toml"),
            os.path.join(path, "env_config.toml")
        )

        if self.normalize_observation or self.normalize_reward:
            self.normalizator.save_stats(path)

    def _simulation_step(self, action):

        # set reinforcement learning agent action
        self.bsk_simulation.apply_rl_agent_action(action)

        # get time horizon
        time_horizon = self.storage.current_sim_secs + self.episode_time_step

        # perform simulation step
        self.bsk_simulation.step(time_horizon)
        bsk_states = self.bsk_simulation.get_simulation_data()

        # check if this is the last step
        if time_horizon >= self.epoch_time_horizon:
            is_last_step = True
        else:
            is_last_step = False

        # store simulation states data
        self.storage.update_records(bsk_states, action, is_last_step)

