import gym
import numpy as np

class MountainCarContinuousObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_obs_points: int = 6):
        super().__init__(env)
        self.max_angle = 1
        
        self.low_state = np.append(self.low_state, -self.max_angle)
        self.high_state = np.append(self.high_state, self.max_angle)
        
        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        
    def observation(self, obs):
        return np.append(obs, np.cos(3 * obs[0]))