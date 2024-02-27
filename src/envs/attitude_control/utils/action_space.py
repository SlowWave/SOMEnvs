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


class ActionSpaceModel():
    def __init__(self):

        # define mapping dictionaries
        self.action_model_map = {
            "1": self._model_1,
            "2": self._model_2,
            "3": self._model_3,
            "4": self._model_4,
        }

        self.action_elaboration_map = {
            "1": self._elaborate_action_1,
            "2": self._elaborate_action_2,
            "3": self._elaborate_action_3,
            "4": self._elaborate_action_4,
        }

        self.model_id = str(CFG["gymnasium"]["action_space"]["model_id"])

    def get_action_space(self):

        return self.action_model_map[self.model_id]()
    
    def get_elaborated_action(self, action, storage):

        return self.action_elaboration_map[self.model_id](action, storage)

    def _model_1(self):

        # define action space limits
        action_limit = np.array(
            [0.5, 0.5, 0.5],
            dtype=np.float32,
        )

        # define action space
        action_space = spaces.Box(
            -action_limit,
            action_limit,
            dtype=np.float32
        )

        return action_space

    def _elaborate_action_1(self, action, storage):

        return action

    def _model_2(self):

        # define action space
        action_space = spaces.MultiDiscrete(
            nvec=np.array([21, 21, 21])
        )

        return action_space

    def _elaborate_action_2(self, action, storage):
        
        # map raw action in the interval [-0.5, 0.5]
        slope = 0.05
        y_intercept = - 0.5

        return action * slope + y_intercept

    def _model_3(self):

       # define action space
       action_space = spaces.MultiDiscrete(
        nvec=np.array([21, 21, 21])
       )

       return action_space

    def _elaborate_action_3(self, action, storage):

        # map raw action in the interval [-0.1, 0.1] 
        slope = 0.01
        y_intercept = - 0.1
        delta_action = action * slope + y_intercept

        # apply delta_torque to past torque control input
        action = storage.actions[-1] + delta_action

        # saturate action
        action = np.clip(action, -0.5, 0.5)

        return action

    def _model_4(self):

        # define action space limits
        action_limit = np.array(
            [0.1, 0.1, 0.1],
            dtype=np.float32,
        )

        # define action space
        action_space = spaces.Box(
            -action_limit,
            action_limit,
            dtype=np.float32
        )

        return action_space

    def _elaborate_action_4(self, action, storage):

        # apply delta_torque to past torque control input
        action = storage.actions[-1] + action

        # define ema window and alpha parameters
        ema_win = 5
        ema_alpha = 0.5   

        # aggregate actions to be filtered
        last_actions = storage.actions[- ema_win:]
        actions_tbf = last_actions + [action]

        # get first value of ema
        ema_values = [actions_tbf[0]]
        
        # apply ema filter
        for i in range(1, len(actions_tbf)):
            ema = ema_alpha * actions_tbf[i] + (1 - ema_alpha) * ema_values[-1]
            ema_values.append(ema)

        action = ema_values[-1]

        # saturate action
        action = np.clip(action, -0.5, 0.5)

        return action

