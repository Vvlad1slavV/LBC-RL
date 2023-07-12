import os
import time

import numpy as np

import ur_control
import ur_control.median_filter as md

np.set_printoptions(precision=4, suppress=True)

J_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

BASE = 'base_link'
TOOL = 'camera_link'
NUM_JOINTS = 6

CONTROL_LOOP_DT = 1.0/100 # 100 hz
ACCELERATION = 0.5

JOINTS = np.array([np.pi/2, -np.pi/2 + np.pi/6 - 0.2, -np.pi/2, -np.pi/2 - np.pi/6 + 0.2, np.pi/2, 0])
print(JOINTS)

median = md.MedianFilter(NUM_JOINTS, 16)

robot_ip = os.getenv('ROBOT_IP', '192.168.88.6')
robot= ur_control.UniversalRobot(robot_ip)
robot_model = ur_control.RobotModel('ur5e_pybullet.urdf', BASE, TOOL)

def ee_local_velocity_2_joint_velocity(ee_vel: np.ndarray):
    robot.update_state()
    ee_base_bel = np.kron(
            np.eye(2),
            robot_model.rot(robot.state.q)
    ) @ ee_vel
    joint_velocities = np.linalg.pinv(robot_model.jacobian(robot.state.q)) @ ee_base_bel

    return joint_velocities


if __name__ == '__main__':
    robot.control.moveJ(JOINTS)
    time.sleep(1.0)

    start_time = time.time()
    while time.time()-start_time < 10.0:
        start_loop_time = time.time()

        action = np.zeros(6)
        action[2] = 0.099*np.sin(start_loop_time - start_time)

        joint_velocity = ee_local_velocity_2_joint_velocity(action)

        robot.control.speedJ(joint_velocity, ACCELERATION, CONTROL_LOOP_DT)

        duration = time.time() - start_loop_time
        if duration < CONTROL_LOOP_DT:
            time.sleep(CONTROL_LOOP_DT - duration)
