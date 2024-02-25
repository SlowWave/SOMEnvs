import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LearningRateCallback(BaseCallback):

    def __init__(
        self,
        initial_lr,
        final_lr,
        total_epochs,
        scheduler_str='linear',
        verbose=0
    ):
        
        super(LearningRateCallback, self).__init__(verbose)

        self.scheduler_map = {
            "linear": self._linear_scheduler,
        }

        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_epochs = total_epochs
        self.scheduler_str = scheduler_str
        self.current_lr = self.initial_lr
        self.current_epoch = 0

    def _linear_scheduler(self):

        if self.current_epoch <= self.total_epochs:
            self.current_lr = self.initial_lr - (self.initial_lr - self.final_lr) * self.current_epoch / self.total_epochs

    def _on_rollout_end(self):

        self.scheduler_map[self.scheduler_str]()
        self.model.learning_rate = self.current_lr
        self.model._setup_lr_schedule()
        self.current_epoch += 1

    def _on_step(self):
        pass
