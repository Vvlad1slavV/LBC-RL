import gym
import numpy as np
import sys

class MountainCarContinuousObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.max_angle = 1

        low_state = np.append(self.low_state, -self.max_angle)
        high_state = np.append(self.high_state, self.max_angle)

        self.observation_space = gym.spaces.Box(
            low=low_state, high=high_state, dtype=np.float32
        )

    def observation(self, obs):
        return np.array([self.state[0], self.state[1], np.cos(3 * self.state[0])])
    
class MountainCarContinuousNoVelObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low_state = self.low_state[0]
        high_state = self.high_state[0]
        self.observation_space = gym.spaces.Box(
            low=low_state, high=high_state, shape=(1,), dtype=np.float32
        )

    def observation(self, obs):
        return np.array(self.state[0], ndmin=1)