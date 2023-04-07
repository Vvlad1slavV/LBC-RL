import os
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for check reward

    :param check_freq: (int)
    :param cpu_num: (int) Size of VecEnv
    :param log_dir: (str) Path to a folder where the evaluations (`.npz`)
        will be saved. It will be updated at each step.
    :param model_name_prefix: (str)
    :param model_name: (str)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self,
                check_freq: int,
                num_cpu: int,
                log_path: str,
                model_save_path: str,
                model_name: str,
                verbose: int = 1,
                ):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.num_cpu = num_cpu
        self.log_path = log_path
        self.model_save_path = model_save_path
        self.model_name = model_name

        self.mean_reward = -np.inf
        self.rew_results = []
        self.mean_reward_ncpu = []
        self.rew_timesteps_ncpu = []
        self.rew_timesteps = []
        self.rew_length = [] # get from ep_info_buffer

        self.ep_len = 0
        self.mean_ep_num = 0

    def _init_callback(self):
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.model_save_path is not None:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def _checkpoint_path(self, extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.model_save_path, f"{self.model_name}_best.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
            x, y = ts2xy(load_results(self.log_path), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last cpu_num episodes
                ep_len = len(y)

                mean_reward_ncpu = np.mean(y[-self.num_cpu:])
                self.mean_reward_ncpu.append(mean_reward_ncpu)
                self.rew_timesteps_ncpu.append(self.num_timesteps)
                self.logger.record("rollout/mean_reward_ncpu", mean_reward_ncpu)
                np.savez(self.model_save_path + self.model_name + '_ncpu',
                         timesteps=self.rew_timesteps_ncpu,
                         results=self.mean_reward_ncpu,
                         ep_lengths=self.rew_length)

                if mean_reward_ncpu > self.mean_reward:
                    model_path = self._checkpoint_path(extension="zip")
                    self.model.save(model_path)
                    self.mean_reward = mean_reward_ncpu

                if self.ep_len != ep_len:
                    self.logger.record("rollout/mean_reward_step", np.mean(y[self.ep_len-ep_len:]))
                    self.ep_len = ep_len
                    self.logger.record("rollout/ep_len", ep_len)
                    if self.ep_len // self.num_cpu != self.mean_ep_num:
                        ep_shift = self.ep_len % self.num_cpu
                        mean_reward = np.mean(y[-self.num_cpu-ep_shift:-1-ep_shift])
                        self.logger.record("rollout/mean_reward", mean_reward)
                        self.rew_results.append(mean_reward)
                        self.rew_timesteps.append(self.num_timesteps)
                        np.savez(self.model_save_path + self.model_name,
                                 timesteps=self.rew_timesteps,
                                 results=self.rew_results,
                                 ep_lengths=self.rew_length)
                        self.mean_ep_num = self.ep_len // self.num_cpu # += 1

        return True
    