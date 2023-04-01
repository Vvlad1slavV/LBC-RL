import gym
from scipy.spatial import distance
import numpy as np
from typing import Callable

class CarRacingGroundTruthObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_obs_points: int = 6):
        super().__init__(env)
        self.num_obs_points = num_obs_points
        self.low_state = np.array(
            [0, -np.pi, *[0, -np.pi]*self.num_obs_points], dtype=np.float32
        )
        self.high_state = np.array(
            [100, np.pi, *[float('inf'), np.pi]*self.num_obs_points], dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        
    def observation(self, obs):
        velocity = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )
        angular_velocity = self.car.hull.angularVelocity
        # ABS sensors
        # abs_sensors = []
        # for i in range(4):
            # abs_sensors.append(0.01 * self.car.wheels[i].omega)
        car_position =  np.array(self.car.hull.position)
        car_angle = -self.car.hull.angle

        track = np.array(self.track)
        track_size = track.shape[0]

        rot = np.array([[np.cos(car_angle), -np.sin(car_angle)], [np.sin(car_angle), np.cos(car_angle)]])
        arg = np.argmin(distance.cdist([car_position], track[:, 2:4]))
        pathxy = rot @ (track[np.r_[arg+1-track_size:arg+1+self.num_obs_points-track_size], 2:4] - car_position).T
        path_angles = np.expand_dims(np.arctan2(*pathxy), axis=0)
        path = np.expand_dims(np.linalg.norm(pathxy, axis=0), axis=0)
        
        return np.array([velocity, angular_velocity, *np.concatenate((path, path_angles), axis=0).ravel(order='F')])

class CarRacingGroundTruthXYObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_obs_points: int = 6):
        super().__init__(env)
        self.num_obs_points = num_obs_points
        self.low_state = np.array(
            [0, -np.pi, *[-float('inf'), -float('inf'), -np.pi]*self.num_obs_points], dtype=np.float32
        )
        self.high_state = np.array(
            [100, np.pi, *[float('inf'), float('inf'), np.pi]*self.num_obs_points], dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        
    def observation(self, obs):
        velocity = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )
        angular_velocity = self.car.hull.angularVelocity
        # ABS sensors
        # abs_sensors = []
        # for i in range(4):
            # abs_sensors.append(0.01 * self.car.wheels[i].omega)
        car_position =  np.array(self.car.hull.position)
        car_angle = -self.car.hull.angle

        track = np.array(self.track)
        track_size = track.shape[0]

        rot = np.array([[np.cos(car_angle), -np.sin(car_angle)], [np.sin(car_angle), np.cos(car_angle)]])
        arg = np.argmin(distance.cdist([car_position], track[:, 2:4]))
        path = rot @ (track[np.r_[arg+1-track_size:arg+1+self.num_obs_points-track_size], 2:4] - car_position).T
        path_angles = np.expand_dims(np.arctan2(*path), axis=0)
        
        return np.array([velocity, angular_velocity, *np.concatenate((path, path_angles), axis=0).ravel(order='F')])

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

def linear_rectified_schedule(initial_value: float, min_value: float = 0.0, *, total_timesteps: int, rectified_timesteps: int) -> Callable[[float], float]:
    """
    Linear learning rate schedule with recified unit.

    :param initial_value: Initial learning rate.
           min_value: Min learning rate.
           total_timesteps: Total timesteps.
           rectified_timesteps: Timesteps after lr will be rectified.
    
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return max(((progress_remaining - 1)*total_timesteps/rectified_timesteps + 1) * initial_value,
                    min_value)

    return func