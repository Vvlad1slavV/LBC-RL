import gym
from scipy.spatial import distance
import numpy as np

class CarRacingGroundTruthObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low_state = np.array(
            [0, -np.pi, *[-float('inf'), -float('inf'), -np.pi]*6], dtype=np.float32
        )
        self.high_state = np.array(
            [100, np.pi, *[float('inf'), float('inf'), np.pi]*6], dtype=np.float32
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
        car_position =  self.car.hull.position
        car_angle = -self.car.hull.angle
        track = np.array(self.track)
        
        rot = np.array([[np.cos(car_angle), -np.sin(car_angle)], [np.sin(car_angle), np.cos(car_angle)]])
        arg = np.argmin(distance.cdist([car_position], track[:, 2:4]))
        track_size = track.shape[0]
        path = rot @ (track[arg+1-track_size:arg+7-track_size:1, 2:4] - car_position).T
        path_angles = np.expand_dims(np.arctan2(*path), axis=0)
        
        # print(path.shape, path_angles.shape)
        # print(np.concatenate((path.T, path_angles.T), axis=1))
        # print()
        # print(path_angles)
        return np.array([velocity, angular_velocity, *np.concatenate((path, path_angles), axis=0).ravel(order='F')])
