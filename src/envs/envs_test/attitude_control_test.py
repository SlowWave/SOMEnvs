import os
import sys

sys.path.append(os.path.normpath(os.path.dirname(__file__) + os.sep + os.pardir))

import time
import numpy as np
from attitude_control.environment import AttitudeControlEnv

env = AttitudeControlEnv()
env.reset()

terminated = False
t1 = time.time()
while not terminated:
    action = env.action_space.sample()
    # action = np.array([0., 0., 0.])
    observation, reward, terminated, truncated, info = env.step(action)
t2 = time.time()

print(t2-t1)

env.plot_results()
