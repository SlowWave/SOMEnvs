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


class DisturbancesGenerator():
    def __init__(self):

        # get constant torques settings
        self.constant_torques_number = CFG["disturbances_generator"]["constant_torques"]["disturbances_number"]
        self.constant_torques_arrays = CFG["disturbances_generator"]["constant_torques"]["disturbances_arrays"]
        self.constant_torques_time_windows = CFG["disturbances_generator"]["constant_torques"]["disturbances_time_windows"]
        self.constant_torques_frames = CFG["disturbances_generator"]["constant_torques"]["disturbances_frames"]

        # get random torques settings
        self.random_torques_number = CFG["disturbances_generator"]["random_torques"]["disturbances_number"]
        self.random_torques_max_amplitudes = CFG["disturbances_generator"]["random_torques"]["max_amplitudes"]
        self.random_torques_min_amplitudes = CFG["disturbances_generator"]["random_torques"]["min_amplitudes"]
        self.random_torques_time_windows = CFG["disturbances_generator"]["random_torques"]["disturbances_time_windows"]

        # get sinusoidal torques settings
        self.sinusoidal_torques_number = CFG["disturbances_generator"]["sinusoidal_torques"]["disturbances_number"]
        self.sinusoidal_torques_arrays = CFG["disturbances_generator"]["sinusoidal_torques"]["disturbances_arrays"]
        self.sinusoidal_torques_frequencies = CFG["disturbances_generator"]["sinusoidal_torques"]["disturbances_frequencies"]
        self.sinusoidal_torques_time_windows = CFG["disturbances_generator"]["sinusoidal_torques"]["disturbances_time_windows"]
        self.sinusoidal_torques_frames = CFG["disturbances_generator"]["sinusoidal_torques"]["disturbances_frames"]

        # set toque disturbances
        self.torques = self._set_disturbances()

    def _set_disturbances(self):

        # define disturbances dictionary
        disturbances = {
            "constant_torques_arrays": list(),
            "constant_torques_time_windows": list(),
            "constant_torques_frames": list(),
            "random_torques_arrays": list(),
            "random_torques_time_windows": list(),
            "sinusoidal_torques_arrays": list(),
            "sinusoidal_torques_time_windows": list(),
            "sinusoidal_torques_frames": list(),
        }

        # aggregate constant disturbances
        for idx in range(self.constant_torques_number):

            disturbances["constant_torques_arrays"].append(self.constant_torques_arrays[idx])
            disturbances["constant_torques_time_windows"].append(self.constant_torques_time_windows[idx])
            disturbances["constant_torques_frames"].append(self.constant_torques_frames[idx])

        # aggregate random disturbances
        for idx in range(self.random_torques_number):

            disturbances["random_torques_arrays"].append(
                np.random.uniform(
                    self.random_torques_min_amplitudes[idx],
                    self.random_torques_max_amplitudes[idx]
                )
            )
            disturbances["random_torques_time_windows"].append(self.random_torques_time_windows[idx])
            
        # aggregate sinusoidal disturbances
        for i in range(self.sinusoidal_torques_number):

            sinusoidal_lambdas = list()

            for j in range(3):

                if self.sinusoidal_torques_frequencies[i][j] > 0:

                    sinusoidal_lambdas.append(
                        lambda t, ii=i, jj=j: self.sinusoidal_torques_arrays[ii][jj] * \
                            np.cos(2 * np.pi * self.sinusoidal_torques_frequencies[ii][jj] * t)
                    )

                else:

                    sinusoidal_lambdas.append(lambda t: 0)

            disturbances["sinusoidal_torques_arrays"].append(sinusoidal_lambdas)
            disturbances["sinusoidal_torques_time_windows"].append(self.sinusoidal_torques_time_windows[i])
            disturbances["sinusoidal_torques_frames"].append(self.sinusoidal_torques_frames[i])

        return disturbances

    def generate_disturbances(self):
        
        return self.torques


