import os
import numpy as np

import ur_control
import ur_control.median_filter as md

np.set_printoptions(precision=4, suppress=True)

J_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

BASE = 'base_link'
TOOL = 'tool0'
NUM_JOINTS = 6

JOINTS  = np.array([np.pi/2, -np.pi/2 + np.pi/6 - 0.2, -np.pi/2, -np.pi/2 - np.pi/6 + 0.2, np.pi/2, 0])
print(JOINTS)

median = md.MedianFilter(NUM_JOINTS, 16)

robot_ip = os.getenv('MONGODB_HOST', '192.168.88.6')
robot= ur_control.UniversalRobot('MONGODB_HOST')
robot.control.moveJ(JOINTS)


