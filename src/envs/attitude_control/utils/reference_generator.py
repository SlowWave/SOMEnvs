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


class ReferenceGenerator():
    def __init__(self):
        
        if CFG["reference_generator"]["use_random_initial_angular_error"]:
            
            max_angular_error = CFG["reference_generator"]["random_initial_angular_errorr_max"]
            min_angular_error = CFG["reference_generator"]["random_initial_angular_errorr_min"]

            self.initial_angular_error = np.random.uniform(min_angular_error, max_angular_error)

        else:

            self.initial_angular_error = CFG["reference_generator"]["initial_angular_error"]

        self.mrp_reference = self._generate_mrp()

    def _generate_mrp(self):

        mrp_reference = [0, 0, 0]

        mrp_square_sum = (1 - np.cos(np.deg2rad(self.initial_angular_error / 2)) ** 2) / \
                            (1 + np.cos(np.deg2rad(self.initial_angular_error / 2))) ** 2

        mrp_reference[0] = (2 * np.random.randint(2) - 1) * np.random.uniform(
            0, np.sqrt(mrp_square_sum)
        )

        mrp_reference[1] = (2 * np.random.randint(2) - 1) * np.random.uniform(
            0, np.sqrt(mrp_square_sum - mrp_reference[0] ** 2)
        )

        mrp_reference[2] = (2 * np.random.randint(2) - 1) * np.sqrt(
            mrp_square_sum - mrp_reference[0] ** 2 - mrp_reference[1] ** 2
        )

        np.random.shuffle(mrp_reference)

        return np.array(mrp_reference)

    def generate_attitude_reference(self):

        return self.mrp_reference


if __name__ == "__main__":

    rg = ReferenceGenerator()
    print(rg.mrp_reference)