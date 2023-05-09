import os
import argparse

import numpy as np
import gym
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import gnwrapper

from CarRacing_utils.schedules import linear_schedule, linear_rectified_schedule
from CarRacing_utils.callbacks import SaveOnBestTrainingRewardCallback

ENV_ID = "CarRacing-v0"

def wrapper(env):
    env = gym.wrappers.gray_scale_observation.GrayScaleObservation(env, keep_dim=True)
    env = gnwrapper.Animation(env)
    return env

def main(opt):
    env = make_vec_env(ENV_ID,
                       n_envs = opt.num_cpu,
                       env_kwargs = { 'verbose': False },
                       wrapper_class = wrapper,
                       monitor_dir = opt.log_prefix + f"logs/{opt.model_name}_monitor/",
                       vec_env_cls = SubprocVecEnv)
    env = VecFrameStack(env, 2)

    callback_list = []

    if opt.model_num_saves != 0:
        callback_list.append(CheckpointCallback(
            save_freq = opt.total_timesteps // opt.model_num_saves // opt.num_cpu,
            save_path = opt.log_prefix + "results/" + opt.model_name,
            name_prefix = opt.model_name,
            save_replay_buffer = False,
            save_vecnormalize = True,
            ))
    
    callback_list.append(SaveOnBestTrainingRewardCallback(
            check_freq = 1024 // opt.num_cpu,
            num_cpu = opt.num_cpu,
            log_path = opt.log_prefix + f"logs/{opt.model_name}_monitor/",
            model_save_path = opt.log_prefix + f"results/{opt.model_name}/",
            model_name = opt.model_name
        ))

    if opt.eval_freq != 0:
        callback_list.append(EvalCallback(env,
                            best_model_save_path= opt.log_prefix + "logs/best_model/" + opt.model_name,
                            # n_eval_episodes = 1,
                            log_path= opt.log_prefix + "results/" + opt.model_name,
                            eval_freq=opt.eval_freq,
                            deterministic=True,
                            render=False))

    policy_kwargs = dict(
                    # activation_fn=nn.GELU,
                    activation_fn=nn.LeakyReLU,
                    net_arch=dict(pi=[256], vf=[256]),
                    ortho_init=False
    )

    model = PPO("CnnPolicy",
            env,
            verbose=1,
            seed=opt.seed,
            batch_size=opt.batch_size,
            # learning_rate=linear_schedule(opt.learning_rate, opt.min_learning_rate),
            learning_rate=linear_rectified_schedule(opt.learning_rate,
                                                    opt.min_learning_rate,
                                                    total_timesteps=opt.total_timesteps,
                                                    rectified_timesteps=opt.rectified_timesteps),
            n_epochs=10,
            n_steps=opt.n_step,
            # clip_range=0.5,
            use_sde=opt.sde,
            # sde_sample_freq=4,
            policy_kwargs=policy_kwargs,
            tensorboard_log = opt.log_prefix + f"logs/{opt.model_name}_tensorboard/",
           )

    model.learn(total_timesteps=opt.total_timesteps, callback=callback_list, progress_bar=True)
    env.close()
    model.save(opt.log_prefix + opt.model_name)

    del model


def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Label smoothing epsilon')
    parser.add_argument('--min-learning-rate', type=float, default=1e-4, help='Label smoothing epsilon')
    parser.add_argument('--sde', action=argparse.BooleanOptionalAction)
    parser.add_argument('--total-timesteps', type=int, default=400_000, help='model.learn total_timesteps')
    parser.add_argument('--rectified-timesteps', type=int, default=400_000, help='model.learn total_timesteps')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--n-step', type=int, default=64, help='PPO n_steps')

    parser.add_argument('--eval-freq', type=int, default=0, help='eval freq')

    parser.add_argument('--model-name', type=str, default='noob', help='model name to save')
    parser.add_argument('--log-prefix', type=str, default='./', help='folder to save logs')
    parser.add_argument('--model-num-saves', type=int, default=10, help='eval freq')

    parser.add_argument('--num-cpu', type=int, default=16, help='Num cpu')

    return parser.parse_known_args()[0] if known else parser.parse_args()    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
