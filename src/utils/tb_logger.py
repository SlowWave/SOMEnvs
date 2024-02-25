import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        
        super(TensorboardCallback, self).__init__(verbose)
        self.is_initialized = False
        self.cumulative_rewards = None

    def _on_rollout_end(self):

        for idx in range(len(self.locals["infos"][0]["rewards"])):
            self.logger.record(
                "rollout/mean_rew_{}".format(idx + 1),
                np.mean(self.cumulative_rewards[idx])
            )
        
        self.is_initialized = False

    def _on_step(self):
        
        if not self.is_initialized:
            self.cumulative_rewards = [0] * len(self.locals["infos"][0]["rewards"])
            self.is_initialized = True
        
        for env_idx in range(len(self.locals["infos"])):
            for idx, reward in enumerate(self.locals["infos"][env_idx]["rewards"]):
                self.cumulative_rewards[idx] += reward / len(self.locals["infos"])

        # for idx in range(len(self.cumulative_rewards)):
        #     self.cumulative_rewards[idx] /= len(self.locals["infos"])

        return True
