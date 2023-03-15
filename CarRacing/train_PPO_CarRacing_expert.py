from typing import Callable
import argparse

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import gnwrapper

from CarRacing_utils import CarRacingGroundTruthObsWrapper

env_id = "CarRacing-v0"
NUM_CPU = 16  # Number of processes to use

def linear_schedule(initial_value: float, min_value: float = 0.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
           min_value: Min learning rate.
    
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return max(progress_remaining * initial_value, min_value)

    return func

def wrapper(env):
    env = CarRacingGroundTruthObsWrapper(env) 
    env = gnwrapper.Animation(env)
    return env


def main(opt):
    env = make_vec_env(env_id, n_envs=NUM_CPU, wrapper_class=wrapper, vec_env_cls=SubprocVecEnv)
    
    eval_callback = EvalCallback(env, 
                             best_model_save_path= opt.log_prefix + "logs/best_model/" + opt.model_name,
                             log_path= opt.log_prefix + "results/" + opt.model_name,
                             eval_freq=500,
                             deterministic=True, render=False)
    
    model = PPO("MlpPolicy",
            env,
            verbose=1,
            seed=opt.seed,
            batch_size=opt.batch_size,
            learning_rate=linear_schedule(1e-3, 1e-5),
            n_epochs=10,
            n_steps=8*NUM_CPU,
            # clip_range=0.5,
            # use_sde=True,
            tensorboard_log= opt.log_prefix + f"logs/{opt.model_name}_tensorboard/",
           )
    
    model.learn(total_timesteps=opt.total_timesteps, callback=eval_callback, progress_bar=True)
    env.close()
    model.save(opt.log_prefix + opt.model_name)
        
    del model
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--total-timesteps', type=int, default=200_000, help='model.learn total_timesteps')
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size for all GPUs')
    parser.add_argument('--model-name', type=str, default='expert', help='model name to save')
    parser.add_argument('--log-prefix', type=str, default='./', help='folder to save logs')
    
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