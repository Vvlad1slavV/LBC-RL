import os
from typing import Callable
from collections import deque
import argparse

import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

import gnwrapper

from CarRacing_utils import CarRacingGroundTruthObsWrapper, CarRacingGroundTruthXYObsWrapper
from CarRacing_utils import linear_schedule

env_id = "CarRacing-v0"

obs_aliases = {
    'DirectionObs': CarRacingGroundTruthObsWrapper,
    'XYObs': CarRacingGroundTruthXYObsWrapper
}

def get_obswrapper_from_name(obswrapper_name: str):
        if obswrapper_name in obs_aliases:
            return obs_aliases[obswrapper_name]
        else:
            raise ValueError(f"Obs Wrapper {obswrapper_name} unknown")
CarObsWrapper = None

def wrapper(env):
    global CarObsWrapper
    env = CarObsWrapper(env) 
    env = gnwrapper.Animation(env)
    return env

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for check reward

    :param check_freq: (int)
    :param cpu_num: (int) Size of VecEnv
    :param log_dir: (str) Path to a folder where the evaluations (`.npz`)
        will be saved. It will be updated at each step.
    :param model_name: (str)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, num_cpu: int, log_path: str, model_name: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.num_cpu = num_cpu
        self.log_path = log_path
        self.model_name = model_name
        self.mean_reward = -np.inf
        self.rew_results = []
        self.rew_timesteps = []
        self.rew_length = [] # get from ep_info_buffer

        self.ep_len = 0
        self.mean_ep_num = 0

    def _init_callback(self):
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.model_name is not None:
            os.makedirs(self.model_name, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
            x, y = ts2xy(load_results(self.log_path), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last cpu_num episodes
                ep_len = len(y)
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
                        np.savez(self.model_name,
                                 timesteps=self.rew_timesteps,
                                 results=self.rew_results,
                                 ep_lengths=self.rew_length)
                        self.mean_ep_num = self.ep_len // self.num_cpu # += 1

        return True

def main(opt):
    global CarObsWrapper
    print(f'Use {opt.wrapper_name} wrapper')
    CarObsWrapper = get_obswrapper_from_name(opt.wrapper_name)
    
    env = make_vec_env(env_id,
                       n_envs = opt.num_cpu,
                       wrapper_class = wrapper,
                       monitor_dir = opt.log_prefix + f"logs/{opt.model_name}_monitor/",
                       vec_env_cls = SubprocVecEnv)
    
    eval_callback = None
    if opt.eval_freq != 0:
        eval_callback = EvalCallback(env,
                                 best_model_save_path= opt.log_prefix + "logs/best_model/" + opt.model_name,
                                 # n_eval_episodes = 1,
                                 log_path= opt.log_prefix + "results/" + opt.model_name,
                                 eval_freq=opt.eval_freq,
                                 deterministic=True, render=False)
    
    model = PPO("MlpPolicy",
            env,
            verbose=1,
            seed=opt.seed,
            batch_size=opt.batch_size,
            learning_rate=linear_schedule(opt.learning_rate, opt.min_learning_rate),
            n_epochs=10,
            n_steps=opt.n_step,
            # clip_range=0.5,
            use_sde=opt.sde,
            tensorboard_log = opt.log_prefix + f"logs/{opt.model_name}_tensorboard/",
           )
    
    print(eval_callback)
    callback = SaveOnBestTrainingRewardCallback(128, opt.num_cpu, opt.log_prefix + f"logs/{opt.model_name}_monitor/", opt.log_prefix + "results/" + opt.model_name + "/rewards")
    model.learn(total_timesteps=opt.total_timesteps, callback=callback, progress_bar=True)
    env.close()
    model.save(opt.log_prefix + opt.model_name)
        
    del model
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Label smoothing epsilon')
    parser.add_argument('--min-learning-rate', type=float, default=1e-4, help='Label smoothing epsilon')
    parser.add_argument('--sde', action=argparse.BooleanOptionalAction)
    parser.add_argument('--total-timesteps', type=int, default=200_000, help='model.learn total_timesteps')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--n-step', type=int, default=64, help='PPO n_steps')
    parser.add_argument('--model-name', type=str, default='expert', help='model name to save')
    parser.add_argument('--log-prefix', type=str, default='./', help='folder to save logs')
    parser.add_argument('--eval-freq', type=int, default=0, help='eval freq')
    parser.add_argument('--wrapper-name', type=str, default='XYObs', help='Name of ObsWrapper')
    
    parser.add_argument('--num-cpu', type=int, default=16, help='Num cpu')

    return parser.parse_known_args()[0] if known else parser.parse_args()    
    
# def run(**kwargs):
#     # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
#     opt = parse_opt(True)
#     for k, v in kwargs.items():
#         setattr(opt, k, v)
#     main(opt)
#     return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)