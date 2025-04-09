#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

def compute_ik(position, orientation):
    """
    Function that calulates the PUMA analytical IK function.
    Should take a 3x1 position vector and a 3x3 rotation matrix,
    and return a list of joint positions.
    """
    
    print("\nReceived Position:")
    print(position)

    print("\nReceived Orientation (3x3 Rotation Matrix):")
    print(orientation)

    # PUMA Robot Parameters (meters)
    d1, a2, a3 = 0.150, 0.432, 0.432  # Given parameters
    
    # Replace with the actual analytical IK computation
    # Insert you code here
    
    # Comment this line when you have your solution
    px, py, pz = position

    # Step 1: Calculate theta1 (shoulder rotation)
    r = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    alpha = np.arctan2(d1, np.sqrt(r**2 - d1**2))
    theta1 = phi - alpha  # Righty solution (add π for lefty)

    # Step 2: Calculate theta3 (elbow joint)
    r_sq = px**2 + py**2 + pz**2 - d1**2
    D = (r_sq - a2**2 - a3**2) / (2 * a2 * a3)
    D = np.clip(D, -1.0, 1.0)  # Ensure D is within valid range for arccos
    theta3 = np.arccos(D)

    # Step 3: Calculate theta2 (shoulder joint)
    s = pz - d1
    phi = np.arctan2(s, r)
    psi = np.arctan2(a3 * np.sin(theta3), a2 + a3 * np.cos(theta3))
    theta2 = phi - psi

    # Step 4: Calculate wrist joint angles (theta4, theta5, theta6)
    def Rz(t):
        return np.array([
            [np.cos(t), -np.sin(t), 0],
            [np.sin(t),  np.cos(t), 0],
            [0,          0,         1]
        ])

    def Ry(t):
        return np.array([
            [np.cos(t),  0, np.sin(t)],
            [0,          1,         0],
            [-np.sin(t), 0, np.cos(t)]
        ])

    # Compute R_03 (rotation matrix for first three joints)
    R_03 = Rz(theta1) @ Ry(theta2) @ Rz(theta3)

    # Compute R_36 (rotation matrix for wrist joints)
    R_36 = R_03.T @ orientation
    # Extract theta5 (ZYX Euler angle beta)
    cos_theta5 = R_36[2, 2]
    cos_theta5 = np.clip(cos_theta5, -1.0, 1.0)
    theta5 = np.arccos(cos_theta5)

    # Handle singularities for theta4 and theta6
    sin_theta5 = np.sqrt(1.0 - cos_theta5**2)
    if abs(sin_theta5) < 1e-6:
        theta4 = 0.0
        theta6 = np.arctan2(R_36[1, 0], R_36[0, 0])
    else:
        theta4 = np.arctan2(R_36[1, 2], R_36[0, 2])
        theta6 = np.arctan2(R_36[2, 1], -R_36[2, 0])

    return np.array([theta1, theta2, theta3, theta4, theta5, theta6])

def pose_callback(msg):
    """
    Callback function to handle incoming end-effector pose messages.
    You probably do not have to change this
    """
    # Extract position (3x1)
    position = np.array([msg.position.x, msg.position.y, msg.position.z])

    # Extract orientation (3x3 rotation matrix from quaternion)
    q = msg.orientation
    orientation = np.array([
        [1 - 2 * (q.y**2 + q.z**2), 2 * (q.x*q.y - q.z*q.w), 2 * (q.x*q.z + q.y*q.w)],
        [2 * (q.x*q.y + q.z*q.w), 1 - 2 * (q.x**2 + q.z**2), 2 * (q.y*q.z - q.x*q.w)],
        [2 * (q.x*q.z - q.y*q.w), 2 * (q.y*q.z + q.x*q.w), 1 - 2 * (q.x**2 + q.y**2)]
    ])

    # Compute inverse kinematics
    joint_positions = compute_ik(position, orientation)

    # Publish joint states
    joint_msg = JointState()
    joint_msg.header.stamp = rospy.Time.now()
    joint_msg.name = [f"joint{i+1}" for i in range(len(joint_positions))]
    joint_msg.position = joint_positions
    joint_pub.publish(joint_msg)

if __name__ == "__main__":
    rospy.init_node("ik_solver_node", anonymous=True)

    # Publisher: sends joint positions
    joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
    
    print("\nWaiting for /goal_pose")
    
    # Subscriber: listens to end-effector pose
    rospy.Subscriber("/goal_pose", Pose, pose_callback)

    rospy.spin()
