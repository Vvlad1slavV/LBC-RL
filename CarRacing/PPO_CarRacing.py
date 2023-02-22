from typing import Callable

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import gnwrapper

from utils import CarRacingGroundTruthObsWrapper

env_id = "CarRacing-v0"
NUM_CPU = 16  # Number0of processes to use

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float, min_value: float) -> float:
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

if __name__ == "__main__":  
    env = make_vec_env(env_id, n_envs=NUM_CPU, wrapper_class=wrapper, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy",
            env,
            verbose=1,
            seed=0,
            batch_size=512,
            learning_rate=linear_schedule(1e-3, 1e-5),
            n_epochs=10,
            n_steps=8*NUM_CPU,
            # clip_range=0.5,
            # use_sde=True,
           )
    
    model.learn(total_timesteps=200_000, progress_bar=True)
    env.close()
    model.save("./ppo_CarRacing_expert")
        
    del model
    
