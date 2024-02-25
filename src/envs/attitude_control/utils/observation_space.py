import os
import tomli
import numpy as np
from gymnasium import spaces


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


class ObservationSpaceModel():
    def __init__(self):

        # define mapping dictionary
        self.observation_model_map = {
            "1": self._model_1,
            "2": self._model_2,
        }


        self.model_id = str(CFG["gymnasium"]["observation_space"]["model_id"])

    def get_observation_space(self):

        return self.observation_model_map[self.model_id]()

    def _model_1(self):

        # define observation space limits
        observation_limit = np.array(
            [
                1,                          # mrp tracking error [0]
                1,                          # mrp tracking error [1]
                1,                          # mrp tracking error [2]
                np.finfo(np.float32).max,   # omega tracking error [0]
                np.finfo(np.float32).max,   # omega tracking error [1]
                np.finfo(np.float32).max,   # omega tracking error [2]
                np.finfo(np.float32).max,   # feedback control signal [0]
                np.finfo(np.float32).max,   # feedback control signal [1]
                np.finfo(np.float32).max,   # feedback control signal [2]
            ],
            dtype=np.float32,
        )

        # define observation space
        observation_space = spaces.Box(
            -observation_limit,
            observation_limit,
            dtype=np.float32
        )

        return observation_space

    def _model_2(self):

        # define observation space limits
        observation_limit = np.array(
            [
                1,                          # mrp tracking error [0]
                1,                          # mrp tracking error [1]
                1,                          # mrp tracking error [2]
                np.finfo(np.float32).max,   # omega tracking error [0]
                np.finfo(np.float32).max,   # omega tracking error [1]
                np.finfo(np.float32).max,   # omega tracking error [2]
                np.finfo(np.float32).max,   # feedback control signal [0]
                np.finfo(np.float32).max,   # feedback control signal [1]
                np.finfo(np.float32).max,   # feedback control signal [2]
                0.5,                        # last rl agent action [0]
                0.5,                        # last rl agent action [1]
                0.5,                        # last rl agent action [2]
            ],
            dtype=np.float32,
        )

        # define observation space
        observation_space = spaces.Box(
            -observation_limit,
            observation_limit,
            dtype=np.float32
        )

        return observation_space

