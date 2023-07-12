import numpy as np
from itmobotics_gym.envs.single_robot_pybullet_env import DefaultValidatingDraft7Validator, SingleRobotPyBulletEnv

from spatialmath import SE3, SO3
from spatialmath import base as sb

import json
import jsonschema
import pkg_resources


class UR5eEnv(SingleRobotPyBulletEnv):
    def __init__(self, config: dict):
        super().__init__(config)

        with open(pkg_resources.resource_filename(__name__, "ur5e_env.json")) as json_file:
            task_schema = json.load(json_file)
            DefaultValidatingDraft7Validator(task_schema).validate(self._env_config)

    def step(self, action: np.ndarray):
        done = False

        # Downgrade reward value depends on time
        reward = -0.001

        self._take_action_vector(action)
        for _ in range(0, int(self._env_config["task"]["control_loop_dt"] / self._sim.time_step)):
            self._sim.sim_step()

        obs = self.observation_state_as_dict()
        if self._sim.sim_time > self._env_config["task"]["termination"]["max_time"]:
            done = True

        info = {
            "sim_time": self._sim.sim_time,
        }
        return obs, reward, done, info
