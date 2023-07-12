import json
import time
from tqdm import tqdm

import numpy as np
from ur5e_env import UR5eEnv

with open("ur5e_env.json", encoding="UTF-8") as json_file:
    env_config = json.load(json_file)

env = UR5eEnv(env_config)
env.seed = int(time.time())
time.sleep(1.0)
env.reset()
time.sleep(1.0)

env._sim.register_objects_for_record()
env._sim.start_record()

pbar = tqdm(total=10)
while True:
    obs = env.observation_state_as_dict()

    sim_time = env._sim.sim_time

    action = np.zeros(6)
    action[2] = 0.099*np.sin(sim_time)

    obs, reward, done, info = env.step(action)
    if done:
        break

    pbar.update(0.02)

env._sim.stop_record()
env._sim.save_scene_record('ur5e.pkl')

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('CLOSED!')

pbar.close()
